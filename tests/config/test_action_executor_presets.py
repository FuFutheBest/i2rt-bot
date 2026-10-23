"""Action-executor preset cleanup contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from manager.utils import (
    AutoActionManagerV2,
    BaseChunkManager,
    JointActionManager,
    SmoothChunkManager,
    load_chunk_manager,
)
from utils.hydra_config import compose_hydra_application_config


CHUNK_MANAGER_PRESETS = [
    ("auto_v2", AutoActionManagerV2),
    ("base", BaseChunkManager),
    ("base_auto_drop", BaseChunkManager),
    ("infer_i2rt_joint", JointActionManager),
    ("robotwin", BaseChunkManager),
    ("smooth", SmoothChunkManager),
]

EEF_EXECUTION_FPS = [
    ("policy_infer/diffusion/config_infer_rizon_eef_dual", 25.0),
    ("policy_infer/diffusion/config_infer_rizon_eef_dual_abs_mode_umi", 25.0),
    ("policy_infer/diffusion/config_infer_rizon_eef_single", 10.0),
]


@pytest.mark.config
@pytest.mark.parametrize(("preset_name", "manager_class"), CHUNK_MANAGER_PRESETS)
def test_canonical_chunk_manager_presets_construct(
    repo_root: Path,
    preset_name: str,
    manager_class: type[BaseChunkManager],
) -> None:
    """Every retained action-executor manager can be constructed from its YAML."""
    config_path = repo_root / "configs" / "action_executor" / f"{preset_name}.yaml"
    executor = yaml.safe_load(config_path.read_text(encoding="utf-8"))["action_executor"]

    manager = load_chunk_manager(executor["chunk_manager"], execution_fps=30.0)

    assert type(manager) is manager_class
    assert manager.execution_fps == 30.0


@pytest.mark.config
def test_default_action_executor_preset_stays_disabled(repo_root: Path) -> None:
    """The default preset is intentionally a no-action-executor profile."""
    config_path = repo_root / "configs" / "action_executor" / "default.yaml"
    executor = yaml.safe_load(config_path.read_text(encoding="utf-8"))["action_executor"]

    assert executor["type"] is None


@pytest.mark.config
@pytest.mark.parametrize(("config_name", "expected_fps"), EEF_EXECUTION_FPS)
def test_eef_configs_preserve_execution_fps(
    repo_root: Path,
    config_name: str,
    expected_fps: float,
) -> None:
    """Base-preset migration retains the legacy EEF control rate."""
    config = compose_hydra_application_config(
        config_name,
        config_dir=repo_root / "configs",
        job_name="action_executor_preset_contract",
    )

    assert config["action_executor"]["execution_fps"] == expected_fps
