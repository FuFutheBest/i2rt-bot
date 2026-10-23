"""Code-owned FastWAM model-variant contracts.

The registry is intentionally independent from Hydra and the model runtime so
configuration UIs can enumerate supported FastWAM variants without importing
Torch or duplicating model-specific fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class FastWAMModelVariant:
    """A native FastWAM configuration and its non-negotiable runtime contract."""

    variant_id: str
    config_name: str
    label: str
    condition_history_frames: int
    action_infer_mode: str
    action_video_attention_mode: str
    required_config_fields: tuple[str, ...]
    required_runtime_fields: tuple[str, ...]


FASTWAM_MODEL_VARIANTS: Mapping[str, FastWAMModelVariant] = MappingProxyType(
    {
        "fastwam": FastWAMModelVariant(
            variant_id="fastwam",
            config_name="policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam",
            label="FastWAM",
            condition_history_frames=1,
            action_infer_mode="observed_frame_cache",
            action_video_attention_mode="first_frame",
            required_config_fields=(
                "condition_history_frames",
                "obs_horizon",
                "action_infer_mode",
                "action_video_attention_mode",
            ),
            required_runtime_fields=(
                "condition_history_frames",
                "action_infer_mode",
                "action_video_attention_mode",
            ),
        ),
        "fastwam-memory": FastWAMModelVariant(
            variant_id="fastwam-memory",
            config_name="policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_memory",
            label="FastWAM memory",
            condition_history_frames=4,
            action_infer_mode="observed_frame_cache",
            action_video_attention_mode="first_frame",
            required_config_fields=(
                "condition_history_frames",
                "obs_horizon",
                "action_infer_mode",
                "action_video_attention_mode",
            ),
            required_runtime_fields=(
                "condition_history_frames",
                "condition_history_stride",
                "action_infer_mode",
                "action_video_attention_mode",
            ),
        ),
        "fasterwam": FastWAMModelVariant(
            variant_id="fasterwam",
            config_name="policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam",
            label="FasterWAM",
            condition_history_frames=1,
            action_infer_mode="one_pass_future_cache",
            action_video_attention_mode="full_video",
            required_config_fields=(
                "condition_history_frames",
                "obs_horizon",
                "action_infer_mode",
                "action_video_attention_mode",
            ),
            required_runtime_fields=(
                "condition_history_frames",
                "action_infer_mode",
                "action_video_attention_mode",
                "sampled_video_num_frames",
            ),
        ),
    }
)


def get_fastwam_model_variant(variant_id: str) -> FastWAMModelVariant:
    """Resolve a supported FastWAM model variant by its stable identifier."""
    normalized = str(variant_id).strip().lower()
    try:
        return FASTWAM_MODEL_VARIANTS[normalized]
    except KeyError as exc:
        raise ValueError(
            "Unknown FastWAM model_variant "
            f"{variant_id!r}; expected one of {sorted(FASTWAM_MODEL_VARIANTS)}"
        ) from exc


def iter_fastwam_model_variants() -> tuple[FastWAMModelVariant, ...]:
    """Return model variants in stable registry order for callers that render choices."""
    return tuple(FASTWAM_MODEL_VARIANTS.values())


def validate_fastwam_variant_config(
    variant_id: str,
    policy_params: Mapping[str, Any],
) -> FastWAMModelVariant:
    """Validate explicit policy parameters against a registered model variant."""
    variant = get_fastwam_model_variant(variant_id)
    configured_variant = policy_params.get("model_variant")
    if configured_variant is not None and str(configured_variant).strip().lower() != variant.variant_id:
        raise ValueError(
            "policy.params.model_variant does not match the selected FastWAM "
            f"variant: {configured_variant!r} != {variant.variant_id!r}"
        )

    expected_values = {
        "condition_history_frames": variant.condition_history_frames,
        "obs_horizon": variant.condition_history_frames,
        "action_infer_mode": variant.action_infer_mode,
        "action_video_attention_mode": variant.action_video_attention_mode,
    }
    missing = [field for field in variant.required_config_fields if field not in policy_params]
    if missing:
        raise ValueError(
            f"FastWAM variant {variant.variant_id!r} requires policy.params "
            f"fields: {', '.join(missing)}"
        )
    for field, expected in expected_values.items():
        value = policy_params[field]
        if field in {"condition_history_frames", "obs_horizon"}:
            matches = int(value) == expected
        else:
            matches = str(value).strip().lower() == expected
        if not matches:
            raise ValueError(
                f"FastWAM variant {variant.variant_id!r} requires "
                f"policy.params.{field}={expected!r}, got {value!r}"
            )
    return variant


def validate_fastwam_variant_runtime_contract(
    variant_id: str,
    runtime_contract: Mapping[str, Any],
) -> FastWAMModelVariant:
    """Check checkpoint-derived server metadata against the selected variant."""
    variant = get_fastwam_model_variant(variant_id)
    missing = [
        field
        for field in variant.required_runtime_fields
        if field not in runtime_contract
    ]
    if missing:
        raise ValueError(
            f"FastWAM runtime contract for {variant.variant_id!r} is missing "
            f"fields: {', '.join(missing)}"
        )
    expected_values = {
        "condition_history_frames": variant.condition_history_frames,
        "action_infer_mode": variant.action_infer_mode,
        "action_video_attention_mode": variant.action_video_attention_mode,
    }
    for field, expected in expected_values.items():
        if field not in runtime_contract:
            raise ValueError(
                f"FastWAM runtime contract for {variant.variant_id!r} is missing {field}"
            )
        value = runtime_contract[field]
        if field == "condition_history_frames":
            matches = int(value) == expected
        else:
            matches = str(value).strip().lower() == expected
        if not matches:
            raise ValueError(
                f"FastWAM variant {variant.variant_id!r} requires checkpoint "
                f"{field}={expected!r}, got {value!r}"
            )
    return variant
