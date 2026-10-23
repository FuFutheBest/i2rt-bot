#!/usr/bin/env python3
"""Download official RoboTwin XPolicyLab cotrain checkpoints."""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPOSITORY = "TianxingChen/RoboTwin2.0"
OFFICIAL_REVISION = "main"
DEFAULT_XPOLICYLAB_ROOT = REPO_ROOT / "third_party" / "RoboTwin" / "XPolicyLab"


@dataclass(frozen=True)
class CheckpointSpec:
    key: str
    policy_name: str
    remote_archive: PurePosixPath
    ckpt_name: str
    env_cfg_type: str = "aloha_agilex"
    action_type: str = "joint"
    seed: int = 0

    @property
    def run_dir_name(self) -> str:
        return f"RoboTwin-{self.ckpt_name}-{self.env_cfg_type}-{self.action_type}-{self.seed}"


SPECS = {
    "pi05": CheckpointSpec(
        key="pi05",
        policy_name="Pi_05",
        remote_archive=PurePosixPath(
            "cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz"
        ),
        ckpt_name="lerobot_v30",
    ),
    "fastwam": CheckpointSpec(
        key="fastwam",
        policy_name="FastWAM",
        remote_archive=PurePosixPath(
            "cotrain_ckpt/FastWAM/RoboTwin-cotrain-aloha_agilex-joint-0.tar.gz"
        ),
        ckpt_name="cotrain",
    ),
}


def _policy_root(xpolicylab_root: Path, spec: CheckpointSpec) -> Path:
    return xpolicylab_root.expanduser().resolve() / "policy" / spec.policy_name


def _checkpoint_root(xpolicylab_root: Path, spec: CheckpointSpec) -> Path:
    return _policy_root(xpolicylab_root, spec) / "checkpoints"


def _destination(xpolicylab_root: Path, spec: CheckpointSpec) -> Path:
    return _checkpoint_root(xpolicylab_root, spec) / spec.run_dir_name


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError(f"Refusing unsafe archive member path: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Refusing archive link member: {member.name}")
        archive.extractall(destination)


def _iter_dirs(root: Path) -> Iterable[Path]:
    yield root
    for path in root.rglob("*"):
        if path.is_dir():
            yield path


def _looks_like_pi05_checkpoint(path: Path) -> bool:
    if (
        (path / "model.safetensors").is_file()
        or (path / "params").exists()
        or (path / "assets").exists()
    ):
        return True
    return any(
        child.is_dir()
        and (
            (child / "model.safetensors").is_file()
            or (child / "params").exists()
            or (child / "assets").exists()
        )
        for child in path.iterdir()
    )


def _looks_like_fastwam_checkpoint(path: Path) -> bool:
    weights_dir = path / "checkpoints" / "weights"
    has_weight = weights_dir.is_dir() and any(weights_dir.glob("step_*.pt"))
    return has_weight and (path / "dataset_stats.json").is_file()


def _checkpoint_predicate(spec: CheckpointSpec) -> Callable[[Path], bool]:
    if spec.key == "pi05":
        return _looks_like_pi05_checkpoint
    if spec.key == "fastwam":
        return _looks_like_fastwam_checkpoint
    raise ValueError(f"Unsupported policy key: {spec.key}")


def _find_extracted_checkpoint(staging: Path, spec: CheckpointSpec) -> Path:
    expected_names = {
        spec.run_dir_name,
        spec.remote_archive.name.removesuffix(".tar.gz"),
    }
    predicate = _checkpoint_predicate(spec)

    for directory in _iter_dirs(staging):
        if directory.name in expected_names and predicate(directory):
            return directory
    matches = [directory for directory in _iter_dirs(staging) if predicate(directory)]
    if len(matches) == 1:
        return matches[0]
    candidates = ", ".join(str(path.relative_to(staging)) for path in matches[:8])
    raise RuntimeError(
        f"Could not identify {spec.policy_name} checkpoint root in extracted archive. "
        f"Matches: {candidates or '<none>'}"
    )


def _install_extracted_checkpoint(source: Path, destination: Path, *, force: bool) -> None:
    if destination.exists():
        if not force:
            print(f"{destination} already exists; use --force to replace it")
            return
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def _download_archive(
    *,
    repository: str,
    revision: str,
    remote_archive: PurePosixPath,
    force: bool,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing huggingface_hub. Install requirements/requirements-robotwin.txt first."
        ) from exc
    return Path(
        hf_hub_download(
            repo_id=repository,
            repo_type="dataset",
            revision=revision,
            filename=remote_archive.as_posix(),
            force_download=force,
        )
    )


def _download_one(
    spec: CheckpointSpec,
    *,
    repository: str,
    revision: str,
    xpolicylab_root: Path,
    force: bool,
    dry_run: bool,
) -> None:
    policy_root = _policy_root(xpolicylab_root, spec)
    if not policy_root.is_dir():
        raise FileNotFoundError(
            f"XPolicyLab policy directory not found: {policy_root}. "
            "Run: git -C third_party/RoboTwin submodule update --init --recursive XPolicyLab"
        )

    destination = _destination(xpolicylab_root, spec)
    print(f"Policy: {spec.policy_name}")
    print(f"Repository: {repository}@{revision}")
    print(f"Archive: {spec.remote_archive}")
    print(f"Destination: {destination}")
    print(f"Launch ckpt name: {spec.ckpt_name}")
    if dry_run:
        return
    if destination.exists() and not force:
        print(f"{destination} already exists; use --force to replace it")
        return

    archive_path = _download_archive(
        repository=repository,
        revision=revision,
        remote_archive=spec.remote_archive,
        force=force,
    )
    checkpoint_root = _checkpoint_root(xpolicylab_root, spec)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{spec.key}-extract-{os.getpid()}-", dir=checkpoint_root
    ) as temporary:
        staging = Path(temporary)
        _safe_extract(archive_path, staging)
        source = _find_extracted_checkpoint(staging, spec)
        _install_extracted_checkpoint(source, destination, force=force)
    print(f"Installed {spec.policy_name} checkpoint at {destination}")


def _selected_specs(policy: str) -> list[CheckpointSpec]:
    if policy == "all":
        return [SPECS["pi05"], SPECS["fastwam"]]
    return [SPECS[policy]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=("pi05", "fastwam", "all"),
        default="all",
        help="Which official XPolicyLab cotrain checkpoint to download",
    )
    parser.add_argument(
        "--repository",
        default=OFFICIAL_REPOSITORY,
        help="Hugging Face dataset repository",
    )
    parser.add_argument(
        "--revision",
        default=OFFICIAL_REVISION,
        help="Hugging Face dataset revision",
    )
    parser.add_argument(
        "--xpolicylab-root",
        type=Path,
        default=DEFAULT_XPOLICYLAB_ROOT,
        help="Local RoboTwin XPolicyLab checkout",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing local checkpoint directories and refresh the HF cache",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print remote and local paths without downloading",
    )
    args = parser.parse_args()

    for spec in _selected_specs(args.policy):
        _download_one(
            spec,
            repository=args.repository,
            revision=args.revision,
            xpolicylab_root=args.xpolicylab_root,
            force=args.force,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
