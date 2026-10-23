from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from utils import openpi_checkpoint
from utils.openpi_checkpoint import (
    OpenPiCheckpointContractError,
    discover_openpi_norm_stats_assets,
    validate_openpi_checkpoint_contract,
)


def _checkpoint(tmp_path: Path, *, asset_id: str = "fixture") -> Path:
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params" / "manifest").write_text("fixture", encoding="utf-8")
    asset_dir = checkpoint / "assets" / asset_id
    asset_dir.mkdir(parents=True)
    (asset_dir / "norm_stats.json").write_text("{}", encoding="utf-8")
    return checkpoint


def _install_openpi(
    monkeypatch: pytest.MonkeyPatch,
    *,
    asset_id: str = "fixture",
    state_dim: int = 16,
    action_dim: int = 16,
) -> None:
    train_config = SimpleNamespace(
        data=SimpleNamespace(
            assets=SimpleNamespace(asset_id=asset_id),
            repo_id="unused",
        )
    )

    def stats(dimension: int) -> Any:
        values = [0.0] * dimension
        return SimpleNamespace(mean=values, std=values, q01=values, q99=values)

    modules = {
        "openpi.training.config": SimpleNamespace(get_config=lambda name: train_config),
        "openpi.training.checkpoints": SimpleNamespace(
            load_norm_stats=lambda assets, selected_asset_id: {
                "state": stats(state_dim),
                "actions": stats(action_dim),
            }
        ),
    }
    original_import = openpi_checkpoint.importlib.import_module
    monkeypatch.setattr(
        openpi_checkpoint.importlib,
        "import_module",
        lambda name: modules[name] if name in modules else original_import(name),
    )


def test_validates_registry_asset_and_dimensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path)
    _install_openpi(monkeypatch)

    contract = validate_openpi_checkpoint_contract(
        "fixture_pi05",
        checkpoint,
        expected_action_dim=16,
        expected_state_dim=16,
    )

    assert contract["asset_id"] == "fixture"
    assert contract["state_dim"] == 16
    assert contract["action_dim"] == 16
    assert contract["checkpoint_dir"] == str(checkpoint.resolve())


def test_discovers_default_or_first_checkpoint_asset(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, asset_id="gift")
    default_dir = checkpoint / "assets" / "default"
    default_dir.mkdir()
    (default_dir / "norm_stats.json").write_text("{}", encoding="utf-8")

    assets = discover_openpi_norm_stats_assets(checkpoint)

    assert assets["asset_id"] == "default"
    assert assets["asset_ids"] == ["default", "gift"]


def test_superoffline_can_override_registry_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path, asset_id="gift")
    _install_openpi(monkeypatch, asset_id="default")

    contract = validate_openpi_checkpoint_contract(
        "default",
        checkpoint,
        norm_stats_asset_id="gift",
        expected_action_dim=16,
        expected_state_dim=16,
    )

    assert contract["config_name"] == "default"
    assert contract["asset_id"] == "gift"


def test_rejects_symlinked_params_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    target = tmp_path / "real-params"
    target.mkdir()
    (target / "manifest").write_text("fixture", encoding="utf-8")
    (checkpoint / "params").symlink_to(target, target_is_directory=True)
    _install_openpi(monkeypatch)

    with pytest.raises(OpenPiCheckpointContractError, match="readable real directory"):
        validate_openpi_checkpoint_contract("fixture_pi05", checkpoint)


def test_rejects_missing_registered_asset_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path, asset_id="default")
    _install_openpi(monkeypatch, asset_id="gift_20260811")

    with pytest.raises(OpenPiCheckpointContractError, match="gift_20260811"):
        validate_openpi_checkpoint_contract("fixture_pi05", checkpoint)


def test_rejects_normalization_dimension_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path)
    _install_openpi(monkeypatch, action_dim=8)

    with pytest.raises(OpenPiCheckpointContractError, match="expected dimension 16"):
        validate_openpi_checkpoint_contract(
            "fixture_pi05", checkpoint, expected_action_dim=16
        )
