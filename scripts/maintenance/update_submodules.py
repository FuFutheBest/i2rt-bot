#!/usr/bin/env python3
"""Refresh README records for configured submodule remote branches."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / ".claude/skills/submodule-maintenance/submodules.json"
)
DEFAULT_README = REPOSITORY_ROOT / "README.md"
TABLE_START = "<!-- SUBMODULE-MAINTENANCE-TABLE:START -->"
TABLE_END = "<!-- SUBMODULE-MAINTENANCE-TABLE:END -->"
TABLE_HEADER = "| Submodule | Tracking branch | Bound commit | Bound date | Latest commit | Latest date |"
TABLE_SEPARATOR = "| --- | --- | --- | --- | --- | --- |"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class UpdateError(RuntimeError):
    """Raised when a configured submodule cannot be inspected safely."""


@dataclass(frozen=True)
class StatusRecord:
    path: str
    branch: str
    bound_commit: str
    bound_date: str
    latest_commit: str
    latest_date: str


def run(*args: str, cwd: Path | None = None, capture_output: bool = False) -> str:
    result = subprocess.run(
        args,
        cwd=cwd or REPOSITORY_ROOT,
        check=False,
        text=True,
        capture_output=capture_output,
    )
    if result.returncode:
        message = result.stderr.strip() if capture_output else ""
        command = " ".join(args)
        raise UpdateError(
            f"Command failed ({result.returncode}): {command}\n{message}".rstrip()
        )
    return result.stdout.strip() if capture_output else ""


def load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise UpdateError(f"Could not read manifest {path}: {error}") from error

    if manifest.get("version") != 1 or not isinstance(manifest.get("submodules"), list):
        raise UpdateError("Manifest must contain version 1 and a submodules array.")

    paths: set[str] = set()
    for entry in manifest["submodules"]:
        required = ("name", "path", "remote", "branch", "enabled")
        if not isinstance(entry, dict) or any(key not in entry for key in required):
            raise UpdateError(f"Invalid submodule entry: {entry!r}")
        if not all(isinstance(entry[key], str) and entry[key] for key in required[:-1]):
            raise UpdateError(
                f"Submodule entry has an empty name, path, remote, or branch: {entry!r}"
            )
        if not isinstance(entry["enabled"], bool):
            raise UpdateError(f"Submodule enabled must be boolean: {entry['name']}")
        if entry["path"].startswith("/") or ".." in Path(entry["path"]).parts:
            raise UpdateError(
                f"Submodule path must stay inside the repository: {entry['path']}"
            )
        if "|" in entry["path"] or "|" in entry["branch"]:
            raise UpdateError(
                f"Submodule path and branch cannot contain a Markdown table separator: {entry['name']}"
            )
        if entry["path"] in paths:
            raise UpdateError(f"Duplicate submodule path: {entry['path']}")
        paths.add(entry["path"])
        if not entry["enabled"] and not entry.get("disabled_reason"):
            raise UpdateError(
                f"Disabled submodule needs disabled_reason: {entry['name']}"
            )
    return manifest["submodules"]


def gitmodules_name(path: str) -> str:
    output = run(
        "git",
        "config",
        "-f",
        ".gitmodules",
        "--get-regexp",
        r"^submodule\..*\.path$",
        capture_output=True,
    )
    for line in output.splitlines():
        key, value = line.split(maxsplit=1)
        if value == path:
            return key.removeprefix("submodule.").removesuffix(".path")
    raise UpdateError(f"Configured path is not a submodule in .gitmodules: {path}")


def gitmodules_value(path: str, key: str) -> str:
    name = gitmodules_name(path)
    return run(
        "git",
        "config",
        "-f",
        ".gitmodules",
        "--get",
        f"submodule.{name}.{key}",
        capture_output=True,
    )


def validate_gitmodules(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        path = entry["path"]
        if gitmodules_value(path, "url") != entry["remote"]:
            raise UpdateError(
                f"Remote mismatch for {path}; update .gitmodules and the manifest together."
            )
        if gitmodules_value(path, "branch") != entry["branch"]:
            raise UpdateError(
                f"Branch mismatch for {path}; update .gitmodules and the manifest together."
            )


def pinned_commit(path: str) -> str:
    output = run("git", "ls-tree", "HEAD", "--", path, capture_output=True)
    fields = output.split()
    if len(fields) < 3 or fields[0] != "160000" or fields[1] != "commit":
        raise UpdateError(f"Configured path is not a gitlink in HEAD: {path}")
    return fields[2]


def ensure_clean(path: Path) -> None:
    if run("git", "status", "--porcelain", cwd=path, capture_output=True):
        raise UpdateError(
            f"Refusing to inspect dirty submodule: {path.relative_to(REPOSITORY_ROOT)}"
        )


def markdown_cell(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    return value


def table_row(record: StatusRecord) -> str:
    return (
        f"| `{record.path}` | `{record.branch}` | `{record.bound_commit}` | {record.bound_date} | "
        f"`{record.latest_commit}` | {record.latest_date} |"
    )


def table_block(records: list[StatusRecord]) -> str:
    lines = [
        TABLE_START,
        TABLE_HEADER,
        TABLE_SEPARATOR,
        *(table_row(record) for record in records),
        TABLE_END,
    ]
    return "\n".join(lines)


def table_bounds(contents: str) -> tuple[int, int]:
    if contents.count(TABLE_START) != 1 or contents.count(TABLE_END) != 1:
        raise UpdateError(
            "README must contain exactly one managed submodule maintenance table."
        )
    start = contents.find(TABLE_START)
    end = contents.find(TABLE_END, start)
    if end < start:
        raise UpdateError(
            "README submodule maintenance table markers are out of order."
        )
    return start, end + len(TABLE_END)


def parse_table_row(line: str) -> StatusRecord:
    if not line.startswith("|") or not line.endswith("|"):
        raise UpdateError(f"Invalid README submodule table row: {line}")
    cells = [markdown_cell(cell) for cell in line[1:-1].split("|")]
    if len(cells) != 6:
        raise UpdateError(f"README submodule table row must have six cells: {line}")
    return StatusRecord(*cells)


def load_status_table(path: Path) -> list[StatusRecord]:
    try:
        contents = path.read_text()
    except OSError as error:
        raise UpdateError(f"Could not read README {path}: {error}") from error

    start, end = table_bounds(contents)
    body = contents[start + len(TABLE_START) : end - len(TABLE_END)]
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if len(lines) < 3 or lines[0] != TABLE_HEADER or lines[1] != TABLE_SEPARATOR:
        raise UpdateError(
            "README submodule maintenance table has an unexpected header."
        )
    return [parse_table_row(line) for line in lines[2:]]


def validate_date(value: str, label: str) -> None:
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise UpdateError(
            f"{label} must be an ISO date (YYYY-MM-DD): {value}"
        ) from error


def validate_status_table(
    entries: list[dict[str, Any]], records: list[StatusRecord]
) -> None:
    paths = [record.path for record in records]
    if len(paths) != len(set(paths)):
        raise UpdateError(
            "README submodule maintenance table contains duplicate paths."
        )

    expected_paths = {entry["path"] for entry in entries}
    if set(paths) != expected_paths:
        raise UpdateError(
            "README submodule maintenance table paths must exactly match the manifest."
        )

    records_by_path = {record.path: record for record in records}
    for entry in entries:
        record = records_by_path[entry["path"]]
        if record.branch != entry["branch"]:
            raise UpdateError(
                f"README tracking branch mismatch for {record.path}; update it with the manifest."
            )
        for label, commit in (
            ("bound commit", record.bound_commit),
            ("latest commit", record.latest_commit),
        ):
            if not COMMIT_PATTERN.fullmatch(commit):
                raise UpdateError(
                    f"README {label} must be a full lowercase Git commit ID for {record.path}: {commit}"
                )
        validate_date(record.bound_date, f"README bound date for {record.path}")
        validate_date(record.latest_date, f"README latest date for {record.path}")


def write_status_table(path: Path, records: list[StatusRecord]) -> None:
    try:
        contents = path.read_text()
    except OSError as error:
        raise UpdateError(f"Could not read README {path}: {error}") from error
    start, end = table_bounds(contents)
    path.write_text(contents[:start] + table_block(records) + contents[end:])


def commit_date(path: Path, commit: str) -> str:
    return run(
        "git",
        "-C",
        str(path),
        "show",
        "-s",
        "--format=%cs",
        f"{commit}^{{commit}}",
        capture_output=True,
    )


def initialize_submodule(path: str) -> Path:
    run("git", "submodule", "sync", "--", path)
    run("git", "submodule", "update", "--init", "--", path)
    module_path = REPOSITORY_ROOT / path
    remotes = run(
        "git", "-C", str(module_path), "remote", capture_output=True
    ).splitlines()
    if "origin" not in remotes:
        run(
            "git",
            "-C",
            str(module_path),
            "remote",
            "add",
            "origin",
            gitmodules_value(path, "url"),
        )
    ensure_clean(module_path)
    return module_path


def refresh_submodule_status(
    entry: dict[str, Any], record: StatusRecord
) -> StatusRecord | None:
    path = entry["path"]
    bound_commit = pinned_commit(path)
    module_path = initialize_submodule(path)
    bound_date = commit_date(module_path, bound_commit)
    if record.bound_commit != bound_commit or record.bound_date != bound_date:
        raise UpdateError(
            f"README binding mismatch for {path}; update the bound commit and date in a manually reviewed "
            "pin-change MR."
        )

    run("git", "-C", str(module_path), "fetch", "--no-tags", "origin", entry["branch"])
    latest_commit = run(
        "git",
        "-C",
        str(module_path),
        "rev-parse",
        "FETCH_HEAD^{commit}",
        capture_output=True,
    )
    latest_date = commit_date(module_path, latest_commit)
    if record.latest_commit == latest_commit and record.latest_date == latest_date:
        return None
    return replace(record, latest_commit=latest_commit, latest_date=latest_date)


def write_report(
    path: Path, updates: list[tuple[dict[str, Any], StatusRecord, StatusRecord]]
) -> None:
    lines = ["# Scheduled Submodule Status Refresh", ""]
    if not updates:
        lines.append("No enabled submodule status records changed.")
    else:
        lines.extend(
            [
                "| Submodule | Branch | Previous latest | Current latest |",
                "| --- | --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| `{entry['path']}` | `{entry['branch']}` | `{old.latest_commit[:12]}` ({old.latest_date}) | "
            f"`{new.latest_commit[:12]}` ({new.latest_date}) |"
            for entry, old, new in updates
        )
        lines.extend(
            [
                "",
                "This automation refreshes README status only; it does not change parent-repository gitlinks "
                "or bound pins.",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the JSON submodule manifest.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=DEFAULT_README,
        help="README containing the maintenance table.",
    )
    parser.add_argument(
        "--report", type=Path, help="Write a Markdown status report to this path."
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate the manifest, .gitmodules, and README table without accessing Git remotes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        entries = load_manifest(args.config)
        records = load_status_table(args.readme)
        validate_gitmodules(entries)
        validate_status_table(entries, records)
        if args.validate_config:
            print(
                f"Validated {len(entries)} submodule entries and the README maintenance table."
            )
            return 0

        records_by_path = {record.path: record for record in records}
        refreshed_records: dict[str, StatusRecord] = {}
        updates: list[tuple[dict[str, Any], StatusRecord, StatusRecord]] = []
        for entry in entries:
            if not entry["enabled"]:
                print(f"Skipping disabled submodule: {entry['path']}")
                continue
            previous = records_by_path[entry["path"]]
            refreshed = refresh_submodule_status(entry, previous)
            if refreshed is None:
                print(f"Already recorded: {entry['path']}")
                continue
            refreshed_records[entry["path"]] = refreshed
            updates.append((entry, previous, refreshed))
            print(
                f"Refreshed {entry['path']}: {previous.latest_commit[:12]} -> {refreshed.latest_commit[:12]}"
            )

        if refreshed_records:
            write_status_table(
                args.readme,
                [refreshed_records.get(record.path, record) for record in records],
            )
        if args.report:
            write_report(args.report, updates)
        return 0
    except UpdateError as error:
        print(f"submodule status refresh failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
