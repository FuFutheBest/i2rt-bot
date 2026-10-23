"""Unit tests for immutable SuperOffline action-calibration bindings."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.superoffline_action_calibration import (
    ActionCalibrationContract,
    ActionCalibrationError,
    prepare_action_calibration,
    snapshot_action_calibrations,
)
from utils.superoffline_registry import get_superoffline_model_contract


def _openpi_contract_and_params() -> tuple[ActionCalibrationContract, dict[str, object]]:
    contract = get_superoffline_model_contract(
        "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi"
    ).action_calibration
    return contract, {
        "tcp_umi_json_path": "calibration/tcp_to_umi_transform.json",
        "gripper_open_max": 0.11,
        "left_gripper_offset": 0.0,
        "right_gripper_offset": 0.0,
    }


def test_unbound_action_calibration_uses_absolute_repository_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract, params = _openpi_contract_and_params()
    monkeypatch.chdir(tmp_path)

    binding, overrides = prepare_action_calibration(contract, params)

    calibration_path = Path(overrides["tcp_umi_json_path"])
    assert binding.calibration_id == "rizon.umi3.tcp-umi.v1"
    assert calibration_path.is_absolute()
    assert calibration_path.name == "tcp_to_umi_transform.json"
    assert calibration_path.is_file()


def test_bound_action_calibration_requires_unchanged_snapshot(tmp_path: Path) -> None:
    contract, params = _openpi_contract_and_params()
    binding, _ = prepare_action_calibration(contract, params)
    snapshot = snapshot_action_calibrations(
        [(contract, binding)],
        tmp_path / "action-calibration",
    )

    _, overrides = prepare_action_calibration(
        contract,
        params,
        binding_id=binding.calibration_id,
        binding_file_digests=binding.file_digests,
        binding_digest=binding.digest,
        snapshot_root=snapshot,
    )
    assert Path(overrides["tcp_umi_json_path"]).parent == snapshot

    (snapshot / "tcp_to_umi").write_text("[]\n", encoding="utf-8")
    with pytest.raises(ActionCalibrationError, match="snapshot"):
        prepare_action_calibration(
            contract,
            params,
            binding_id=binding.calibration_id,
            binding_file_digests=binding.file_digests,
            binding_digest=binding.digest,
            snapshot_root=snapshot,
        )
