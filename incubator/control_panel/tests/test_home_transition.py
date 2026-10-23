"""Hardware-free checks for sampled whole-episode home targets."""

from __future__ import annotations

import pytest

from incubator.control_panel.home_transition import (
    HomeTransitionError,
    normalize_home_target,
    select_candidate_home_target,
)


def _target(candidate_id: str = "left-01") -> dict:
    return {
        "schema_version": 1,
        "source": "home_pose_candidates",
        "config_name": "demo",
        "config_hash": "a" * 64,
        "arms": {
            "left": {
                "candidate_id": candidate_id,
                "joints": [0.1] * 7,
                "collision_checked": False,
            },
            "right": {
                "candidate_id": "right-01",
                "joints": [-0.1] * 7,
                "collision_checked": False,
            },
        },
    }


def test_candidate_selection_uses_one_current_candidate_per_arm() -> None:
    state = {
        "status": "ready",
        "config_name": "demo",
        "config_hash": "b" * 64,
        "arms": {
            "left": {
                "candidates": [
                    {"id": "left-01", "joints": [0.1] * 7},
                    {"id": "left-02", "joints": [0.2] * 7},
                ]
            },
            "right": {
                "candidates": [
                    {"id": "right-01", "joints": [-0.1] * 7},
                    {"id": "right-02", "joints": [-0.2] * 7},
                ]
            },
        },
    }

    selected = select_candidate_home_target(
        state,
        expected_config="demo",
        chooser=lambda candidates: candidates[-1],
    )

    assert selected is not None
    assert selected["arms"]["left"]["candidate_id"] == "left-02"
    assert selected["arms"]["right"]["candidate_id"] == "right-02"
    assert selected["config_hash"] == "b" * 64


def test_candidate_selection_accepts_namespaced_hydra_config() -> None:
    state = {
        "status": "ready",
        "config_name": "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
        "config_hash": "b" * 64,
        "arms": {
            side: {"candidates": [{"id": f"{side}-01", "joints": [0.0] * 7}]}
            for side in ("left", "right")
        },
    }

    selected = select_candidate_home_target(
        state,
        expected_config="policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
    )

    assert selected is not None
    assert selected["config_name"] == state["config_name"]


def test_candidate_selection_refuses_stale_or_partial_arm_state() -> None:
    assert (
        select_candidate_home_target(
            {"status": "ready", "config_name": "other"},
            expected_config="demo",
        )
        is None
    )
    assert (
        select_candidate_home_target(
            {
                "status": "partial",
                "config_name": "demo",
                "config_hash": "a" * 64,
                "arms": {
                    "left": {"candidates": []},
                    "right": {"candidates": []},
                },
            },
            expected_config="demo",
        )
        is None
    )


def test_candidate_selection_is_stable_for_idempotent_retry() -> None:
    state = {
        "status": "ready",
        "config_name": "demo",
        "config_hash": "c" * 64,
        "arms": {
            side: {
                "candidates": [
                    {"id": f"{side}-{index}", "joints": [float(index)] * 7}
                    for index in range(6)
                ]
            }
            for side in ("left", "right")
        },
    }

    first = select_candidate_home_target(
        state,
        expected_config="demo",
        selection_key="request-123",
    )
    retry = select_candidate_home_target(
        state,
        expected_config="demo",
        selection_key="request-123",
    )

    assert first == retry


def test_home_target_validation_rejects_non_finite_or_incomplete_joints() -> None:
    target = _target()
    target["arms"]["left"]["joints"] = [0.0] * 6 + [float("nan")]
    with pytest.raises(HomeTransitionError, match="7 finite"):
        normalize_home_target(target)
