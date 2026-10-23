#!/usr/bin/env python3
"""Download an official RoboTwin joint14 ACT checkpoint into the bridge layout."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPOSITORY = "TianxingChen/RoboTwin2.0"
# The dataset's mutable main branch no longer retains these ACT artifacts.
# This official commit contains the matched policy and normalization statistics.
OFFICIAL_REVISION = "790104e1bbe9f70574b8c77d3676f229e6f2daeb"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "robotwin" / "act"
ARTIFACT_NAMES = ("policy_last.ckpt", "dataset_stats.pkl")


def checkpoint_layout(
    *,
    checkpoint_root: Path,
    task: str,
    task_config: str,
    expert_data_num: int,
) -> tuple[Path, dict[str, PurePosixPath]]:
    """Map the official ACT dataset tree to SuperInference's configured layout."""
    if not task or not task_config:
        raise ValueError("task and task_config must not be empty")
    if expert_data_num <= 0:
        raise ValueError("expert_data_num must be positive")
    setting = f"{task_config}-{expert_data_num}"
    destination = checkpoint_root.expanduser().resolve() / task / setting
    remote_directory = PurePosixPath("act_ckpt") / f"act-{task}" / setting
    return destination, {name: remote_directory / name for name in ARTIFACT_NAMES}


def _copy_atomically(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        default=OFFICIAL_REPOSITORY,
        help="Hugging Face dataset repository",
    )
    parser.add_argument(
        "--revision",
        default=OFFICIAL_REVISION,
        help="Hugging Face dataset revision; default pins the matching official ACT pair",
    )
    parser.add_argument("--task", default="beat_block_hammer")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--expert-data-num", type=int, default=50)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing local checkpoint artifacts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print remote and local paths without downloading",
    )
    args = parser.parse_args()

    destination, remote_paths = checkpoint_layout(
        checkpoint_root=args.checkpoint_root,
        task=args.task,
        task_config=args.task_config,
        expert_data_num=args.expert_data_num,
    )
    print(f"Repository: {args.repository}@{args.revision}")
    print(f"Destination: {destination}")
    for artifact_name, remote_path in remote_paths.items():
        print(f"{remote_path} -> {destination / artifact_name}")
    if args.dry_run:
        return 0

    targets = [destination / artifact_name for artifact_name in ARTIFACT_NAMES]
    existing = [target.exists() for target in targets]
    if any(existing) and not args.force:
        if all(existing):
            print("ACT checkpoint artifacts already exist; use --force to replace them")
            return 0
        raise FileExistsError(
            f"ACT checkpoint destination is incomplete: {destination}. "
            "Remove it or pass --force to replace the artifacts."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing huggingface_hub. Install third_party/RoboTwin/scripts/requirements.txt first."
        ) from exc

    cached_paths = {
        artifact_name: Path(
            hf_hub_download(
                repo_id=args.repository,
                repo_type="dataset",
                revision=args.revision,
                filename=remote_path.as_posix(),
                force_download=args.force,
            )
        )
        for artifact_name, remote_path in remote_paths.items()
    }
    for artifact_name in ARTIFACT_NAMES:
        cached_path = cached_paths[artifact_name]
        _copy_atomically(Path(cached_path), destination / artifact_name)
        print(f"Downloaded {artifact_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
