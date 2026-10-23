"""FastWAM model-variant registry contracts without model runtime imports."""

from __future__ import annotations

import pytest

from consumer.inference.fastwam_model_registry import (
    FASTWAM_MODEL_VARIANTS,
    get_fastwam_model_variant,
    validate_fastwam_variant_config,
    validate_fastwam_variant_runtime_contract,
)


def test_registry_exposes_native_fastwam_variants() -> None:
    assert tuple(FASTWAM_MODEL_VARIANTS) == (
        "fastwam",
        "fastwam-memory",
        "fasterwam",
    )
    assert get_fastwam_model_variant("fasterwam").config_name == (
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam"
    )


@pytest.mark.parametrize(
    ("variant_id", "params"),
    (
        (
            "fastwam-memory",
            {
                "model_variant": "fastwam-memory",
                "condition_history_frames": 4,
                "obs_horizon": 4,
                "action_infer_mode": "observed_frame_cache",
                "action_video_attention_mode": "first_frame",
            },
        ),
        (
            "fasterwam",
            {
                "model_variant": "fasterwam",
                "condition_history_frames": 1,
                "obs_horizon": 1,
                "action_infer_mode": "one_pass_future_cache",
                "action_video_attention_mode": "full_video",
            },
        ),
    ),
)
def test_registry_config_validation_leaves_checkpoint_derived_fields_optional(
    variant_id: str,
    params: dict[str, object],
) -> None:
    variant = validate_fastwam_variant_config(variant_id, params)

    assert variant.variant_id == variant_id


def test_fasterwam_runtime_contract_requires_checkpoint_sampling_metadata() -> None:
    with pytest.raises(ValueError, match="sampled_video_num_frames"):
        validate_fastwam_variant_runtime_contract(
            "fasterwam",
            {
                "condition_history_frames": 1,
                "action_infer_mode": "one_pass_future_cache",
                "action_video_attention_mode": "full_video",
            },
        )
