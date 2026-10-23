#!/usr/bin/env python3
"""Identify an installed Flexiv Elements Studio build from local vendor archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile


ARCHIVE_PATTERN = re.compile(r"FlexivElementsStudio_v(?P<version>[^/]+)\.zip$", re.IGNORECASE)
ARCHIVE_MEMBER = "FlexivElementsStudio/FlexivElementsStudio"


def _sha256_stream(stream: object) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _installed_executable(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    result = subprocess.run(
        ("pgrep", "-n", "-f", r"/FlexivElementsStudio( |$)"),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    executable = Path("/proc") / result.stdout.strip() / "exe"
    return executable.resolve() if executable.exists() else None


def _archives(roots: list[Path]) -> list[Path]:
    candidates: set[Path] = set()
    for root in roots:
        expanded = root.expanduser()
        if expanded.is_file():
            candidates.add(expanded.resolve())
        elif expanded.is_dir():
            candidates.update(path.resolve() for path in expanded.rglob("FlexivElementsStudio_v*.zip"))
    return sorted(candidates)


def _archive_digest(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as archive, archive.open(ARCHIVE_MEMBER) as member:
            return _sha256_stream(member)
    except (KeyError, OSError, zipfile.BadZipFile):
        return None


def _cached_robot_versions(install_root: Path) -> list[dict[str, str]]:
    license_root = install_root / "user_data_ui" / "License"
    robots: list[dict[str, str]] = []
    for path in sorted(license_root.glob("featureID*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            serial = payload["serial_number"][0]["Arm serial number"]
            software = payload["software_version"][0]["RobotControlApp"]
        except (KeyError, IndexError, OSError, TypeError, json.JSONDecodeError):
            continue
        robots.append({"serial": str(serial), "robot_software": str(software), "source": str(path)})
    return robots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, help="Installed FlexivElementsStudio executable")
    parser.add_argument(
        "--archive-root",
        action="append",
        type=Path,
        dest="archive_roots",
        help="Archive file or directory to search recursively; repeat as needed",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    executable = _installed_executable(args.executable)
    roots = args.archive_roots or [Path.home() / "Downloads", Path.home() / "\u4e0b\u8f7d"]
    payload: dict[str, object] = {
        "executable": str(executable) if executable else None,
        "archive_roots": [str(root.expanduser()) for root in roots],
    }
    if executable is None or not executable.is_file():
        payload.update(status="not_found", detail="Elements executable is not running; pass --executable")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    with executable.open("rb") as installed:
        installed_digest = _sha256_stream(installed)
    archives = _archives(roots)
    matches: list[dict[str, str]] = []
    for archive in archives:
        match = ARCHIVE_PATTERN.search(archive.name)
        digest = _archive_digest(archive)
        if match and digest == installed_digest:
            matches.append({"version": match.group("version"), "archive": str(archive)})

    versions = sorted({item["version"] for item in matches})
    payload.update(
        installed_sha256=installed_digest,
        archives_scanned=len(archives),
        matches=matches,
        version=versions[0] if len(versions) == 1 else None,
        cached_robots=_cached_robot_versions(executable.parent),
    )
    if not matches:
        payload.update(status="unmatched", detail="no archive executable matches the installed binary")
        status = 3
    elif len(versions) > 1:
        payload.update(status="ambiguous", detail="matching archives claim different versions")
        status = 4
    else:
        payload.update(status="matched")
        status = 0

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Elements executable: {executable}")
        print(f"Installed SHA-256: {installed_digest}")
        print(f"Archives scanned: {len(archives)}")
        for item in matches:
            print(f"MATCH v{item['version']}: {item['archive']}")
        for robot in payload["cached_robots"]:
            print(f"CACHED ROBOT {robot['serial']}: {robot['robot_software']} ({robot['source']})")
        print(f"Status: {payload['status']}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
