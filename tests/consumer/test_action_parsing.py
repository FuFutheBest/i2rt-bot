"""Policy action parsing contract tests.

Author: OpenAI Assistant
"""

from __future__ import annotations

import numpy as np
import pytest

from consumer.inference.openpi_policy_infer import (
    _OpenPiPolicyInferMixin,
    _config_file_path,
    _repository_root,
)


@pytest.fixture
def openpi_parser() -> object:
    """Create an OpenPI parser without running its initializer."""
    parser = _OpenPiPolicyInferMixin.__new__(_OpenPiPolicyInferMixin)
    parser.summary_shm = None
    parser.running = False
    return parser


@pytest.mark.unit
def test_single_arm_action_chunk_parses_xyz_quaternion_gripper(
    openpi_parser: object,
    single_arm_action_chunk: np.ndarray,
) -> None:
    """Single-arm action chunks use [xyz, xyzw quaternion, gripper]."""
    parsed = openpi_parser._parse_single_arm_action(single_arm_action_chunk)

    assert set(parsed) == {"single_arm"}
    np.testing.assert_array_equal(parsed["single_arm"]["position"], single_arm_action_chunk[:, 0:3])
    np.testing.assert_array_equal(parsed["single_arm"]["quaternion"], single_arm_action_chunk[:, 3:7])
    np.testing.assert_array_equal(parsed["single_arm"]["gripper"], single_arm_action_chunk[:, 7:8])


@pytest.mark.unit
def test_dual_arm_action_chunk_parses_left_and_right_targets(
    openpi_parser: object,
    dual_arm_action_chunk: np.ndarray,
) -> None:
    """Dual-arm action chunks are two concatenated single-arm layouts."""
    parsed = openpi_parser._parse_dual_arm_action(dual_arm_action_chunk)

    assert set(parsed) == {"left_arm", "right_arm"}
    np.testing.assert_array_equal(parsed["left_arm"]["position"], dual_arm_action_chunk[:, 0:3])
    np.testing.assert_array_equal(parsed["left_arm"]["quaternion"], dual_arm_action_chunk[:, 3:7])
    np.testing.assert_array_equal(parsed["left_arm"]["gripper"], dual_arm_action_chunk[:, 7:8])
    np.testing.assert_array_equal(parsed["right_arm"]["position"], dual_arm_action_chunk[:, 8:11])
    np.testing.assert_array_equal(parsed["right_arm"]["quaternion"], dual_arm_action_chunk[:, 11:15])
    np.testing.assert_array_equal(parsed["right_arm"]["gripper"], dual_arm_action_chunk[:, 15:16])


@pytest.mark.unit
def test_invalid_action_width_returns_empty_mapping(openpi_parser: object) -> None:
    """Unexpected action dimensions should not produce partially parsed actions."""
    invalid_chunk = np.zeros((2, 7), dtype=np.float64)

    assert openpi_parser._parse_single_arm_action(invalid_chunk) == {}
    assert openpi_parser._parse_dual_arm_action(invalid_chunk) == {}


@pytest.mark.unit
def test_openpi_entrypoint_resolves_the_repository_root() -> None:
    root = _repository_root()

    assert (root / "consumer" / "inference" / "openpi_policy_infer.py").is_file()
    assert (root / "configs").is_dir()


@pytest.mark.unit
def test_openpi_entrypoint_resolves_nested_hydra_config(tmp_path) -> None:
    config = tmp_path / "configs" / "policy_infer" / "openpi" / "demo.yaml"
    config.parent.mkdir(parents=True)
    config.touch()

    assert _config_file_path("policy_infer/openpi/demo", repo_root=tmp_path) == str(config)
    assert _config_file_path("policy_infer/openpi/demo.yaml", repo_root=tmp_path) == str(config)
    assert _config_file_path("policy_infer/openpi/missing", repo_root=tmp_path) is None
