"""Discover and bind machine-local Wan model assets for FastWAM."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from omegaconf import OmegaConf

from utils.hydra_config import compose_hydra_application_config

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
REDIRECT_MODEL_ID = "DiffSynth-Studio/Wan-Series-Converted-Safetensors"
LOCAL_STATE_DIR = Path(".superinference/local")
DEFAULT_CACHE_DIR = LOCAL_STATE_DIR / "model-cache"
DEFAULT_MANIFEST = LOCAL_STATE_DIR / "fastwam_model_sources.json"

_WAN_MODEL_PATTERNS = (
    "Wan2.2_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "diffusion_pytorch_model*.safetensors",
)
_REDIRECT_PATTERNS = (
    "Wan2.2_VAE.safetensors",
    "models_t5_umt5-xxl-enc-bf16.safetensors",
)
_TOKENIZER_PATTERN = "google/umt5-xxl/"
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "spiece.model",
)
_PRUNED_DIRECTORY_NAMES = {
    ".git",
    ".cache",
    ".conda",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".venv",
    ".venvsuperinf",
    "Trash",
    "__pycache__",
    "anaconda3",
    "data",
    "datasets",
    "env",
    "logs",
    "miniconda3",
    "node_modules",
    "outputs",
    "tmp",
    "venv",
}


class FastWAMModelCacheError(RuntimeError):
    """Raised when a local FastWAM model cache cannot be prepared safely."""


@dataclass(frozen=True)
class ModelAssetSpec:
    model_id: str
    required_patterns: tuple[str, ...]
    known_patterns: tuple[str, ...]
    target: Path | None
    direct_source: Path | None = None


@dataclass(frozen=True)
class ModelCachePlan:
    config_name: str
    cache_root: Path
    manifest_path: Path
    assets: tuple[ModelAssetSpec, ...]
    checkpoint_root: Path | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelCandidate:
    path: Path
    origin: str
    completeness: int


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return bool(value)


def _checkpoint_root(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser().resolve(strict=False)
    if path.is_file():
        if path.parent.name == "weights" and path.parent.parent.name == "checkpoints":
            return path.parent.parent.parent
        return path.parent
    return path if path.is_dir() else None


def _training_fastwam_config(checkpoint_root: Path | None) -> dict[str, Any]:
    if checkpoint_root is None:
        return {}
    config_path = checkpoint_root / "config.yaml"
    if not config_path.is_file():
        return {}
    config = OmegaConf.load(config_path)
    value = OmegaConf.select(config, "framework.fastwam", default={})
    return dict(OmegaConf.to_container(value, resolve=True) or {})


def _remote_model_id(value: str) -> str:
    model_id = value.strip().strip("/")
    path = PurePosixPath(model_id)
    if not model_id or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise FastWAMModelCacheError(f"invalid remote model id: {value!r}")
    return path.as_posix()


def _append_asset(
    assets: dict[str, dict[str, Any]],
    *,
    model_id: str,
    pattern: str,
    known_patterns: tuple[str, ...],
    cache_root: Path,
) -> None:
    expanded = Path(model_id).expanduser()
    if expanded.is_absolute() or expanded.exists():
        source = expanded.resolve(strict=False)
        key = str(source)
        entry = assets.setdefault(
            key,
            {
                "model_id": str(source),
                "required": [],
                "known": known_patterns,
                "target": None,
                "direct_source": source,
            },
        )
    else:
        remote_id = _remote_model_id(model_id)
        entry = assets.setdefault(
            remote_id,
            {
                "model_id": remote_id,
                "required": [],
                "known": known_patterns,
                "target": cache_root / PurePosixPath(remote_id),
                "direct_source": None,
            },
        )
    if pattern not in entry["required"]:
        entry["required"].append(pattern)


def build_model_cache_plan(
    config_name: str,
    *,
    repo_root: Path,
) -> ModelCachePlan:
    """Resolve the effective Wan assets required by one FastWAM config."""
    root = repo_root.expanduser().resolve()
    config = compose_hydra_application_config(
        config_name,
        config_dir=root / "configs",
        job_name="fastwam_model_cache",
    )
    policy = config.get("policy", {})
    if not isinstance(policy, Mapping):
        raise FastWAMModelCacheError("config policy section must be a mapping")
    params = dict(policy.get("params", {}) or {})
    server = params.pop("server", {}) or {}
    if isinstance(server, Mapping):
        params.update(server)
    if str(policy.get("class", "")) != "FastWAMPolicy":
        raise FastWAMModelCacheError(
            f"config {config_name!r} is not a direct FastWAM policy"
        )

    configured_cache = Path(str(params.get("model_base_path", DEFAULT_CACHE_DIR)))
    cache_root = (
        configured_cache.expanduser().resolve(strict=False)
        if configured_cache.is_absolute()
        else (root / configured_cache).resolve(strict=False)
    )
    local_root = (root / LOCAL_STATE_DIR).resolve(strict=False)
    if not cache_root.is_relative_to(local_root):
        raise FastWAMModelCacheError(
            "FastWAM model cache must remain below .superinference/local: "
            f"{cache_root}"
        )

    checkpoint_root = _checkpoint_root(params.get("ckpt_path"))
    training = _training_fastwam_config(checkpoint_root)
    warnings: list[str] = []
    if checkpoint_root is None:
        warnings.append(
            "checkpoint is unavailable; using the reviewed default Wan model ids"
        )
    elif not (checkpoint_root / "config.yaml").is_file():
        warnings.append(
            f"checkpoint training config is unavailable: {checkpoint_root / 'config.yaml'}"
        )

    model_id = str(params.get("model_id", training.get("model_id", DEFAULT_MODEL_ID)))
    tokenizer_model_id = str(
        params.get(
            "tokenizer_model_id",
            training.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID),
        )
    )
    load_text_encoder = _as_bool(
        params.get("load_text_encoder", training.get("load_text_encoder", False))
    )
    skip_dit = _as_bool(params.get("skip_pretrain_backbone_load", True))
    redirect_common = _as_bool(training.get("redirect_common_files", False))
    explicit_text = bool(params.get("text_encoder_path"))
    explicit_tokenizer = bool(params.get("tokenizer_path"))
    if explicit_text != explicit_tokenizer:
        raise FastWAMModelCacheError(
            "text_encoder_path and tokenizer_path must be configured together"
        )

    assets: dict[str, dict[str, Any]] = {}
    if not skip_dit:
        _append_asset(
            assets,
            model_id=model_id,
            pattern="diffusion_pytorch_model*.safetensors",
            known_patterns=_WAN_MODEL_PATTERNS,
            cache_root=cache_root,
        )
    common_model_id = REDIRECT_MODEL_ID if redirect_common else model_id
    vae_pattern = "Wan2.2_VAE.safetensors" if redirect_common else "Wan2.2_VAE.pth"
    _append_asset(
        assets,
        model_id=common_model_id,
        pattern=vae_pattern,
        known_patterns=_REDIRECT_PATTERNS if redirect_common else _WAN_MODEL_PATTERNS,
        cache_root=cache_root,
    )
    if load_text_encoder and not explicit_text:
        text_pattern = (
            "models_t5_umt5-xxl-enc-bf16.safetensors"
            if redirect_common
            else "models_t5_umt5-xxl-enc-bf16.pth"
        )
        _append_asset(
            assets,
            model_id=common_model_id,
            pattern=text_pattern,
            known_patterns=_REDIRECT_PATTERNS if redirect_common else _WAN_MODEL_PATTERNS,
            cache_root=cache_root,
        )
        _append_asset(
            assets,
            model_id=tokenizer_model_id,
            pattern=_TOKENIZER_PATTERN,
            known_patterns=(_TOKENIZER_PATTERN,),
            cache_root=cache_root,
        )

    specs = tuple(
        ModelAssetSpec(
            model_id=str(entry["model_id"]),
            required_patterns=tuple(entry["required"]),
            known_patterns=tuple(entry["known"]),
            target=entry["target"],
            direct_source=entry["direct_source"],
        )
        for entry in assets.values()
    )
    return ModelCachePlan(
        config_name=config_name,
        cache_root=cache_root,
        manifest_path=root / DEFAULT_MANIFEST,
        assets=specs,
        checkpoint_root=checkpoint_root,
        warnings=tuple(warnings),
    )


def _pattern_matches(root: Path, pattern: str) -> bool:
    if pattern.endswith("/"):
        directory = root / pattern.rstrip("/")
        if not directory.is_dir():
            return False
        if pattern == _TOKENIZER_PATTERN:
            return all((directory / name).is_file() for name in _TOKENIZER_FILES)
        return any(directory.iterdir())
    return any(root.glob(pattern))


def validate_source(path: Path, patterns: Iterable[str]) -> tuple[bool, tuple[str, ...]]:
    root = path.expanduser().resolve(strict=False)
    missing = tuple(pattern for pattern in patterns if not _pattern_matches(root, pattern))
    return not missing, missing


def _modelscope_candidates(home: Path, model_id: str, environment: Mapping[str, str]) -> Iterable[Path]:
    organization, _, name = model_id.partition("/")
    if not organization or not name:
        return ()
    roots = [home / ".cache/modelscope"]
    configured = environment.get("MODELSCOPE_CACHE")
    if configured:
        roots.insert(0, Path(configured).expanduser())
    candidates: list[Path] = []
    for root in roots:
        candidates.extend((root / "models" / f"{organization}--{name}" / "snapshots").glob("*"))
        candidates.append(root / "hub" / organization / name)
        candidates.append(root / organization / name)
    return candidates


def _huggingface_candidates(home: Path, model_id: str, environment: Mapping[str, str]) -> Iterable[Path]:
    organization, _, name = model_id.partition("/")
    if not organization or not name:
        return ()
    roots: list[Path] = []
    if environment.get("HUGGINGFACE_HUB_CACHE"):
        roots.append(Path(environment["HUGGINGFACE_HUB_CACHE"]).expanduser())
    hf_home = Path(environment.get("HF_HOME", home / ".cache/huggingface")).expanduser()
    roots.append(hf_home / "hub")
    candidates: list[Path] = []
    for root in roots:
        candidates.extend((root / f"models--{organization}--{name}" / "snapshots").glob("*"))
    return candidates


def _bounded_home_candidates(
    home: Path,
    basename: str,
    *,
    max_depth: int,
    excluded_roots: tuple[Path, ...],
) -> Iterable[Path]:
    home = home.resolve(strict=False)
    for current, directories, _files in os.walk(home, followlinks=False):
        current_path = Path(current)
        try:
            depth = len(current_path.relative_to(home).parts)
        except ValueError:
            directories[:] = []
            continue
        if depth >= max_depth:
            directories[:] = []
            continue
        kept: list[str] = []
        for name in directories:
            child = current_path / name
            if name in _PRUNED_DIRECTORY_NAMES or name.startswith("pytest-of-"):
                continue
            if any(child == root or child.is_relative_to(root) for root in excluded_roots):
                continue
            if name == basename:
                yield child
                continue
            kept.append(name)
        directories[:] = kept


def discover_candidates(
    spec: ModelAssetSpec,
    *,
    home: Path,
    repo_root: Path,
    environment: Mapping[str, str] | None = None,
    max_depth: int = 5,
) -> tuple[ModelCandidate, ...]:
    """Return validated candidates, preferring the most complete source."""
    env = dict(os.environ if environment is None else environment)
    if spec.direct_source is not None:
        valid, _missing = validate_source(spec.direct_source, spec.required_patterns)
        return (
            ModelCandidate(spec.direct_source, "configured-local-path", len(spec.known_patterns)),
        ) if valid else ()

    assert spec.target is not None
    model_id = _remote_model_id(spec.model_id)
    basename = PurePosixPath(model_id).name
    candidates: list[tuple[Path, str, int]] = [(spec.target, "configured-cache", 0)]

    base_path = env.get("DIFFSYNTH_MODEL_BASE_PATH")
    if base_path:
        candidates.append((Path(base_path).expanduser() / PurePosixPath(model_id), "environment", 5))
    for path in _modelscope_candidates(home, model_id, env):
        candidates.append((path, "modelscope-cache", 10))
    for path in _huggingface_candidates(home, model_id, env):
        candidates.append((path, "huggingface-cache", 15))

    conventional_roots = (
        home / "models",
        home / "model",
        home / "checkpoints",
        home / "code",
        home / "code/wan",
        home / "wan",
        home / "workspace",
        home / ".local/share/superinference/checkpoints",
    )
    relative_id = Path(*PurePosixPath(model_id).parts)
    for root in conventional_roots:
        candidates.append((root / relative_id, "conventional-root", 20))
        candidates.append((root / basename, "conventional-root", 21))

    excluded = (
        (repo_root / LOCAL_STATE_DIR).resolve(strict=False),
        (home / ".local/share/Trash").resolve(strict=False),
    )
    for path in _bounded_home_candidates(
        home,
        basename,
        max_depth=max_depth,
        excluded_roots=excluded,
    ):
        candidates.append((path, "bounded-home-scan", 50))

    discovered: list[tuple[ModelCandidate, int]] = []
    seen: set[Path] = set()
    for path, origin, priority in candidates:
        if not path.exists():
            continue
        resolved = path.expanduser().resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        valid, _missing = validate_source(resolved, spec.required_patterns)
        if not valid:
            continue
        completeness = sum(
            1 for pattern in spec.known_patterns if _pattern_matches(resolved, pattern)
        )
        discovered.append((ModelCandidate(resolved, origin, completeness), priority))
    discovered.sort(key=lambda item: (-item[0].completeness, item[1], len(str(item[0].path))))
    return tuple(item[0] for item in discovered)


def _quarantine_target(target: Path, cache_root: Path) -> Path | None:
    if not target.exists() and not target.is_symlink():
        return None
    if target.is_symlink():
        target.unlink()
        return None
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    relative = target.relative_to(cache_root)
    destination = cache_root / ".incomplete" / f"{timestamp}-{os.getpid()}" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(target, destination)
    return destination


def _bind_target(target: Path, source: Path, cache_root: Path) -> Path | None:
    quarantined = _quarantine_target(target, cache_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.bind-{uuid.uuid4().hex}"
    temporary.symlink_to(source, target_is_directory=True)
    os.replace(temporary, target)
    return quarantined


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare_model_cache(
    plan: ModelCachePlan,
    *,
    repo_root: Path,
    home: Path,
    environment: Mapping[str, str] | None = None,
    max_depth: int = 5,
    apply: bool = True,
    discover_unbound: bool = True,
) -> dict[str, Any]:
    """Discover complete local assets and optionally bind them into the cache."""
    results: list[dict[str, Any]] = []
    all_ready = True
    for spec in plan.assets:
        if spec.direct_source is not None:
            valid, missing = validate_source(spec.direct_source, spec.required_patterns)
            all_ready = all_ready and valid
            results.append(
                {
                    "model_id": spec.model_id,
                    "status": "direct" if valid else "missing",
                    "source": str(spec.direct_source),
                    "target": None,
                    "missing": list(missing),
                }
            )
            continue

        assert spec.target is not None
        target_valid, target_missing = validate_source(spec.target, spec.required_patterns)
        if target_valid:
            results.append(
                {
                    "model_id": spec.model_id,
                    "status": "ready",
                    "source": str(spec.target.resolve(strict=False)),
                    "target": str(spec.target),
                    "missing": [],
                }
            )
            continue

        if not discover_unbound:
            all_ready = False
            results.append(
                {
                    "model_id": spec.model_id,
                    "status": "invalid",
                    "source": None,
                    "target": str(spec.target),
                    "missing": list(target_missing or spec.required_patterns),
                }
            )
            continue

        candidates = discover_candidates(
            spec,
            home=home,
            repo_root=repo_root,
            environment=environment,
            max_depth=max_depth,
        )
        if not candidates:
            all_ready = False
            results.append(
                {
                    "model_id": spec.model_id,
                    "status": "download-required",
                    "source": None,
                    "target": str(spec.target),
                    "missing": list(target_missing or spec.required_patterns),
                }
            )
            continue
        selected = candidates[0]
        quarantined = None
        if apply:
            quarantined = _bind_target(spec.target, selected.path, plan.cache_root)
        results.append(
            {
                "model_id": spec.model_id,
                "status": "bound" if apply else "discovered",
                "source": str(selected.path),
                "source_origin": selected.origin,
                "source_completeness": selected.completeness,
                "target": str(spec.target),
                "quarantined": None if quarantined is None else str(quarantined),
                "missing": [],
            }
        )

    payload = {
        "schema": "superinference.fastwam-model-sources.v1",
        "config_name": plan.config_name,
        "cache_root": str(plan.cache_root),
        "checkpoint_root": None if plan.checkpoint_root is None else str(plan.checkpoint_root),
        "ready": all_ready,
        "warnings": list(plan.warnings),
        "assets": results,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if apply:
        _atomic_write_json(plan.manifest_path, payload)
    return payload


__all__ = [
    "FastWAMModelCacheError",
    "ModelAssetSpec",
    "ModelCachePlan",
    "build_model_cache_plan",
    "discover_candidates",
    "prepare_model_cache",
    "validate_source",
]
