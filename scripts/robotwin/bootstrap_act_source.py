#!/usr/bin/env python3
"""Export RoboTwin's historical standalone ACT package without changing its checkout."""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOTWIN_ROOT = REPO_ROOT / "third_party" / "RoboTwin"
DEFAULT_DESTINATION = REPO_ROOT / "third_party" / "robotwin_act" / "policy"


def _safe_extract_act(archive: bytes, destination: Path) -> int:
    """Extract only policy/ACT files, rejecting path traversal in the git archive."""
    file_count = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        for member in bundle.getmembers():
            path = PurePosixPath(member.name)
            if not path.parts or path.parts[:2] != ("policy", "ACT"):
                continue
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe ACT archive entry: {member.name}")
            # git archive emits the policy/ACT directory itself before its
            # regular files. Directories carry no payload and are safe to skip.
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Unsafe ACT archive entry: {member.name}")
            relative = Path(*path.parts[1:])
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read ACT archive entry: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            file_count += 1
    if file_count == 0:
        raise ValueError("The requested git ref does not contain policy/ACT")
    return file_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, default=DEFAULT_ROBOTWIN_ROOT)
    parser.add_argument(
        "--ref",
        default="origin/stable_2.0",
        help="RoboTwin git ref containing policy/ACT",
    )
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing destination ACT package",
    )
    args = parser.parse_args()

    robotwin_root = args.robotwin_root.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    act_destination = destination / "ACT"
    if not (robotwin_root / ".git").exists():
        raise FileNotFoundError(f"RoboTwin git checkout not found: {robotwin_root}")
    if act_destination.exists():
        if not args.force:
            raise FileExistsError(
                f"ACT source already exists: {act_destination}; use --force to replace it"
            )
        if not act_destination.is_dir():
            raise ValueError(
                f"Refusing to replace non-directory ACT destination: {act_destination}"
            )
        shutil.rmtree(act_destination)

    result = subprocess.run(
        [
            "git",
            "-C",
            str(robotwin_root),
            "archive",
            "--format=tar",
            args.ref,
            "policy/ACT",
        ],
        check=True,
        capture_output=True,
    )
    count = _safe_extract_act(result.stdout, destination)
    print(f"Exported {count} ACT files to {act_destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
