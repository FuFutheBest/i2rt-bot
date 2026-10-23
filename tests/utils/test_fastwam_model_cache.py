from __future__ import annotations

from pathlib import Path

import pytest

from utils.fastwam_model_cache import (
    ModelAssetSpec,
    ModelCachePlan,
    build_model_cache_plan,
    discover_candidates,
    prepare_model_cache,
)

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
TOKENIZER_ID = "Wan-AI/Wan2.1-T2V-1.3B"
FASTWAM_CONFIGS = (
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam",
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_rtc",
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_memory",
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam",
)


def _write_model(root: Path, *, with_dit: bool) -> Path:
    root.mkdir(parents=True)
    (root / "Wan2.2_VAE.pth").write_bytes(b"vae")
    (root / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"text")
    if with_dit:
        (root / "diffusion_pytorch_model-00001-of-00003.safetensors").write_bytes(
            b"dit"
        )
    return root


def _write_tokenizer(root: Path) -> Path:
    tokenizer = root / "google/umt5-xxl"
    tokenizer.mkdir(parents=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "spiece.model"):
        (tokenizer / name).write_text("{}", encoding="utf-8")
    return root


@pytest.mark.config
@pytest.mark.parametrize("config_name", FASTWAM_CONFIGS)
def test_all_direct_fastwam_configs_resolve_portable_model_cache_plan(
    repo_root: Path,
    config_name: str,
) -> None:
    plan = build_model_cache_plan(config_name, repo_root=repo_root)

    assets = {asset.model_id: asset for asset in plan.assets}
    assert set(assets) == {MODEL_ID, TOKENIZER_ID}
    assert assets[MODEL_ID].required_patterns == (
        "Wan2.2_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
    )
    assert assets[TOKENIZER_ID].required_patterns == ("google/umt5-xxl/",)
    assert plan.cache_root == (
        repo_root / ".superinference/local/model-cache"
    ).resolve()


def test_discovery_prefers_more_complete_conventional_model(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    modelscope = _write_model(
        home
        / ".cache/modelscope/models/Wan-AI--Wan2.2-TI2V-5B/snapshots/master",
        with_dit=False,
    )
    complete = _write_model(home / "code/wan/Wan2.2-TI2V-5B", with_dit=True)
    target = tmp_path / "repo/.superinference/local/model-cache/Wan-AI/Wan2.2-TI2V-5B"
    spec = ModelAssetSpec(
        model_id=MODEL_ID,
        required_patterns=("Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth"),
        known_patterns=(
            "Wan2.2_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "diffusion_pytorch_model*.safetensors",
        ),
        target=target,
    )

    candidates = discover_candidates(
        spec,
        home=home,
        repo_root=tmp_path / "repo",
        environment={},
    )

    assert candidates[0].path == complete.resolve()
    assert candidates[0].completeness == 3
    assert modelscope.resolve() in {candidate.path for candidate in candidates}


def test_discovery_ignores_models_in_trash(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_model(
        home / ".local/share/Trash/files/Wan2.2-TI2V-5B",
        with_dit=True,
    )
    spec = ModelAssetSpec(
        model_id=MODEL_ID,
        required_patterns=("Wan2.2_VAE.pth",),
        known_patterns=("Wan2.2_VAE.pth",),
        target=tmp_path
        / "repo/.superinference/local/model-cache/Wan-AI/Wan2.2-TI2V-5B",
    )

    candidates = discover_candidates(
        spec,
        home=home,
        repo_root=tmp_path / "repo",
        environment={},
    )

    assert candidates == ()


def test_prepare_binds_assets_and_quarantines_incomplete_target(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    home = tmp_path / "home"
    cache = repo / ".superinference/local/model-cache"
    model_source = _write_model(home / "code/wan/Wan2.2-TI2V-5B", with_dit=True)
    tokenizer_source = _write_tokenizer(home / "code/Wan2.1-T2V-1.3B")
    model_target = cache / "Wan-AI/Wan2.2-TI2V-5B"
    model_target.mkdir(parents=True)
    (model_target / "._____temp").mkdir()
    (model_target / "._____temp/Wan2.2_VAE.pth").write_bytes(b"partial")
    tokenizer_target = cache / "Wan-AI/Wan2.1-T2V-1.3B"
    plan = ModelCachePlan(
        config_name="policy_infer/fastwam/test",
        cache_root=cache,
        manifest_path=repo / ".superinference/local/fastwam_model_sources.json",
        assets=(
            ModelAssetSpec(
                model_id=MODEL_ID,
                required_patterns=("Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth"),
                known_patterns=(
                    "Wan2.2_VAE.pth",
                    "models_t5_umt5-xxl-enc-bf16.pth",
                    "diffusion_pytorch_model*.safetensors",
                ),
                target=model_target,
            ),
            ModelAssetSpec(
                model_id=TOKENIZER_ID,
                required_patterns=("google/umt5-xxl/",),
                known_patterns=("google/umt5-xxl/",),
                target=tokenizer_target,
            ),
        ),
        checkpoint_root=None,
    )

    payload = prepare_model_cache(
        plan,
        repo_root=repo,
        home=home,
        environment={},
    )

    assert payload["ready"] is True
    assert model_target.is_symlink()
    assert model_target.resolve() == model_source.resolve()
    assert tokenizer_target.is_symlink()
    assert tokenizer_target.resolve() == tokenizer_source.resolve()
    assert any((cache / ".incomplete").rglob("Wan2.2_VAE.pth"))
    assert plan.manifest_path.is_file()


def test_prepare_reports_download_when_no_complete_source_exists(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    cache = repo / ".superinference/local/model-cache"
    target = cache / "Wan-AI/Wan2.2-TI2V-5B"
    plan = ModelCachePlan(
        config_name="policy_infer/fastwam/test",
        cache_root=cache,
        manifest_path=repo / ".superinference/local/fastwam_model_sources.json",
        assets=(
            ModelAssetSpec(
                model_id=MODEL_ID,
                required_patterns=("Wan2.2_VAE.pth",),
                known_patterns=("Wan2.2_VAE.pth",),
                target=target,
            ),
        ),
        checkpoint_root=None,
    )

    payload = prepare_model_cache(
        plan,
        repo_root=repo,
        home=tmp_path / "home",
        environment={},
    )

    assert payload["ready"] is False
    assert payload["assets"][0]["status"] == "download-required"
    assert not target.exists()


def test_doctor_mode_requires_the_configured_cache_target(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    home = tmp_path / "home"
    _write_model(home / "code/wan/Wan2.2-TI2V-5B", with_dit=True)
    cache = repo / ".superinference/local/model-cache"
    target = cache / "Wan-AI/Wan2.2-TI2V-5B"
    plan = ModelCachePlan(
        config_name="policy_infer/fastwam/test",
        cache_root=cache,
        manifest_path=repo / ".superinference/local/fastwam_model_sources.json",
        assets=(
            ModelAssetSpec(
                model_id=MODEL_ID,
                required_patterns=("Wan2.2_VAE.pth",),
                known_patterns=("Wan2.2_VAE.pth",),
                target=target,
            ),
        ),
        checkpoint_root=None,
    )

    payload = prepare_model_cache(
        plan,
        repo_root=repo,
        home=home,
        environment={},
        apply=False,
        discover_unbound=False,
    )

    assert payload["ready"] is False
    assert payload["assets"][0]["status"] == "invalid"
