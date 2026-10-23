"""Runtime-only path derivation for inference recordings."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Optional


RECORDING_DATE_DIRECTORY = re.compile(r"^[0-9]{8}$")


def configured_recording_root(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> Optional[Path]:
    """Resolve the YAML-owned rollout root without changing the config."""
    policy = config.get("policy")
    rollout = policy.get("infer_rollout") if isinstance(policy, Mapping) else None
    raw_root = rollout.get("output_dir") if isinstance(rollout, Mapping) else None
    if not isinstance(raw_root, str) or not raw_root.strip():
        return None
    root = Path(raw_root.strip()).expanduser()
    if not root.is_absolute():
        root = Path(repo_root) / root
    return root.resolve(strict=False)


def iter_recording_episode_roots(root: Path) -> Iterator[Path]:
    """Yield the configured root and legacy YYYYMMDD children for replay."""
    root = Path(root).expanduser().resolve(strict=False)
    yield root
    if not root.is_dir() or root.is_symlink():
        return
    try:
        children = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError:
        return
    for child in children:
        if RECORDING_DATE_DIRECTORY.fullmatch(child.name) is None:
            continue
        try:
            if child.is_symlink() or not child.is_dir():
                continue
        except OSError:
            continue
        yield child


def recording_root_has_episodes(root: Path) -> bool:
    """Return whether a flat or dated root contains at least one episode."""
    for episode_root in iter_recording_episode_roots(root):
        if not episode_root.is_dir():
            continue
        try:
            children = episode_root.iterdir()
        except OSError:
            continue
        for child in children:
            try:
                if (
                    child.name.startswith("episode_")
                    and not child.is_symlink()
                    and child.is_dir()
                ):
                    return True
            except OSError:
                continue
    return False


def _recording_root_modified_ns(root: Path) -> int:
    modified_ns = 0
    for episode_root in iter_recording_episode_roots(root):
        try:
            modified_ns = max(modified_ns, int(episode_root.stat().st_mtime_ns))
        except OSError:
            continue
    return modified_ns


def latest_sibling_recording_root(
    configured_root: Path,
    *,
    sibling_parents: Iterable[Path] = (),
) -> Path:
    """Resolve the newest populated copy or sibling of a configured dataset."""
    configured_root = Path(configured_root).expanduser().resolve(strict=False)
    parents: list[Path] = []
    for raw_parent in (configured_root.parent, *sibling_parents):
        parent = Path(raw_parent).expanduser().resolve(strict=False)
        if parent not in parents:
            parents.append(parent)

    exact_candidates: list[Path] = []
    for parent in parents:
        candidate = parent / configured_root.name
        if recording_root_has_episodes(candidate):
            exact_candidates.append(candidate)
    if exact_candidates:
        return max(
            exact_candidates,
            key=lambda item: (_recording_root_modified_ns(item), str(item)),
        ).resolve(strict=False)

    latest: Optional[tuple[int, str, Path]] = None
    for parent in parents:
        if not parent.is_dir() or parent.is_symlink():
            continue
        try:
            candidates = parent.iterdir()
        except OSError:
            continue
        for candidate in candidates:
            try:
                if candidate.is_symlink() or not candidate.is_dir():
                    continue
                if not recording_root_has_episodes(candidate):
                    continue
                modified_ns = _recording_root_modified_ns(candidate)
            except OSError:
                continue
            key = (modified_ns, candidate.name, candidate)
            if latest is None or key[:2] > latest[:2]:
                latest = key
    return latest[2] if latest is not None else configured_root


__all__ = [
    "RECORDING_DATE_DIRECTORY",
    "configured_recording_root",
    "iter_recording_episode_roots",
    "latest_sibling_recording_root",
    "recording_root_has_episodes",
]
