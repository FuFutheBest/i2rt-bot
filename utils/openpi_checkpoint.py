"""Validate the OpenPI registry and checkpoint assets before model loading."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class OpenPiCheckpointContractError(ValueError):
    """The selected OpenPI registry entry and checkpoint do not agree."""


def discover_openpi_norm_stats_assets(
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Inspect one canonical OpenPI checkpoint without importing OpenPI."""
    candidate = Path(checkpoint_dir).expanduser()
    if candidate.is_symlink():
        raise OpenPiCheckpointContractError(
            f"Checkpoint directory must be the canonical directory, not a symlink: {candidate}"
        )
    checkpoint = candidate.resolve(strict=False)
    if not checkpoint.is_dir() or not os.access(checkpoint, os.R_OK | os.X_OK):
        raise OpenPiCheckpointContractError(
            f"Checkpoint directory does not exist or is not readable: {checkpoint}"
        )

    params_dir = checkpoint / "params"
    if params_dir.is_symlink() or not params_dir.is_dir() or not os.access(params_dir, os.R_OK | os.X_OK):
        raise OpenPiCheckpointContractError(
            f"Checkpoint params must be a readable real directory: {params_dir}"
        )
    try:
        next(params_dir.iterdir())
    except StopIteration as exc:
        raise OpenPiCheckpointContractError(
            f"Checkpoint params directory is empty: {params_dir}"
        ) from exc

    assets: dict[str, str] = {}
    assets_dir = checkpoint / "assets"
    if assets_dir.is_dir():
        for norm_stats_path in sorted(assets_dir.glob("*/norm_stats.json")):
            if (
                norm_stats_path.is_symlink()
                or not norm_stats_path.is_file()
                or not os.access(norm_stats_path, os.R_OK)
                or norm_stats_path.stat().st_size <= 0
            ):
                continue
            assets[norm_stats_path.parent.name] = str(norm_stats_path)
    if not assets:
        raise OpenPiCheckpointContractError(
            f"OpenPI checkpoint has no readable assets/*/norm_stats.json: {checkpoint}"
        )
    selected_asset_id = "default" if "default" in assets else min(assets)
    return {
        "checkpoint_dir": str(checkpoint),
        "params_dir": str(params_dir),
        "asset_id": selected_asset_id,
        "asset_ids": sorted(assets),
        "norm_stats_path": assets[selected_asset_id],
    }


def _vector_dimension(stats: Any, key: str) -> int:
    lengths: dict[str, int] = {}
    for field in ("mean", "std", "q01", "q99"):
        value = getattr(stats, field, None)
        if value is None:
            continue
        shape = getattr(value, "shape", None)
        if shape is not None:
            if len(shape) != 1:
                raise OpenPiCheckpointContractError(
                    f"Normalization stats '{key}.{field}' must be one-dimensional; got shape {tuple(shape)}"
                )
            length = int(shape[0])
        else:
            try:
                length = len(value)
            except TypeError as exc:
                raise OpenPiCheckpointContractError(
                    f"Normalization stats '{key}.{field}' is not a vector"
                ) from exc
        lengths[field] = length

    if "mean" not in lengths or "std" not in lengths:
        raise OpenPiCheckpointContractError(
            f"Normalization stats '{key}' must contain mean and std vectors"
        )
    if len(set(lengths.values())) != 1:
        raise OpenPiCheckpointContractError(
            f"Normalization stats '{key}' has inconsistent vector dimensions: {lengths}"
        )
    dimension = lengths["mean"]
    if dimension <= 0:
        raise OpenPiCheckpointContractError(
            f"Normalization stats '{key}' must not be empty"
        )
    return dimension


def _asset_id(train_config: Any) -> str:
    data_factory = getattr(train_config, "data", None)
    assets = getattr(data_factory, "assets", None)
    asset_id = getattr(assets, "asset_id", None)
    if not isinstance(asset_id, str) or not asset_id.strip():
        asset_id = getattr(data_factory, "repo_id", None)
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise OpenPiCheckpointContractError(
            "OpenPI registry config does not define data.assets.asset_id or data.repo_id"
        )
    return asset_id


def validate_openpi_checkpoint_contract(
    pi_config_name: str,
    checkpoint_dir: str | Path,
    *,
    norm_stats_asset_id: str | None = None,
    expected_action_dim: int | None = None,
    expected_state_dim: int | None = None,
) -> dict[str, Any]:
    """Validate the files and normalization contract used by OpenPI inference.

    OpenPI imports stay lazy so device-only Python profiles can import this module.
    The caller must activate the intended OpenPI source before invoking this function.
    """
    if not isinstance(pi_config_name, str) or not pi_config_name.strip():
        raise OpenPiCheckpointContractError("OpenPI registry config name is empty")

    discovered = discover_openpi_norm_stats_assets(checkpoint_dir)
    checkpoint = Path(discovered["checkpoint_dir"])
    params_dir = Path(discovered["params_dir"])

    try:
        config_module = importlib.import_module("openpi.training.config")
        checkpoints_module = importlib.import_module("openpi.training.checkpoints")
        train_config = config_module.get_config(pi_config_name)
    except Exception as exc:
        raise OpenPiCheckpointContractError(
            f"Cannot resolve OpenPI registry config '{pi_config_name}': {exc}"
        ) from exc

    asset_id = (
        str(norm_stats_asset_id).strip()
        if norm_stats_asset_id is not None
        else _asset_id(train_config)
    )
    if not asset_id:
        raise OpenPiCheckpointContractError("OpenPI norm-stats asset id is empty")
    asset_dir = checkpoint / "assets" / asset_id
    norm_stats_path = asset_dir / "norm_stats.json"
    if asset_dir.is_symlink() or not asset_dir.is_dir():
        raise OpenPiCheckpointContractError(
            f"Checkpoint asset directory for registry asset_id '{asset_id}' does not exist: {asset_dir}"
        )
    if (
        norm_stats_path.is_symlink()
        or not norm_stats_path.is_file()
        or not os.access(norm_stats_path, os.R_OK)
        or norm_stats_path.stat().st_size <= 0
    ):
        raise OpenPiCheckpointContractError(
            f"Checkpoint norm stats for registry asset_id '{asset_id}' do not exist or are not readable: "
            f"{norm_stats_path}"
        )

    try:
        norm_stats = checkpoints_module.load_norm_stats(checkpoint / "assets", asset_id)
    except Exception as exc:
        raise OpenPiCheckpointContractError(
            f"Cannot load OpenPI norm stats from {norm_stats_path}: {exc}"
        ) from exc
    if not isinstance(norm_stats, Mapping):
        raise OpenPiCheckpointContractError(
            f"OpenPI norm stats must be a mapping: {norm_stats_path}"
        )

    dimensions: dict[str, int] = {}
    for key in ("state", "actions"):
        if key not in norm_stats:
            raise OpenPiCheckpointContractError(
                f"OpenPI norm stats are missing '{key}': {norm_stats_path}"
            )
        dimensions[key] = _vector_dimension(norm_stats[key], key)

    expected_dimensions = {
        "actions": expected_action_dim,
        "state": expected_state_dim,
    }
    for key, expected_dimension in expected_dimensions.items():
        if expected_dimension is None:
            continue
        if expected_dimension <= 0:
            raise OpenPiCheckpointContractError(
                f"Expected OpenPI {key} dimension must be positive; got {expected_dimension}"
            )
        if dimensions[key] != expected_dimension:
            raise OpenPiCheckpointContractError(
                f"OpenPI normalization '{key}' dimension {dimensions[key]} does not match "
                f"the expected dimension {expected_dimension}"
            )

    return {
        "config_name": pi_config_name,
        "checkpoint_dir": str(checkpoint),
        "params_dir": str(params_dir),
        "asset_id": asset_id,
        "norm_stats_path": str(norm_stats_path),
        "state_dim": dimensions["state"],
        "action_dim": dimensions["actions"],
    }
