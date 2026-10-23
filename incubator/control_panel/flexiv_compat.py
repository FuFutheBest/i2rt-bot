"""Read-only discovery of installed Flexiv Elements software.

Elements is useful diagnostic context, but it is not an RDK compatibility
source. Robot firmware compatibility is resolved separately by
``incubator.control_panel.flexiv_rdk``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Iterable, Mapping, Optional, Sequence

from .hardware_discovery import default_elements_cache_paths


ELEMENTS_VERSION_ENV = "SUPERINFERENCE_ELEMENTS_VERSION"
ELEMENTS_DEFAULT_ENV = "SUPERINFERENCE_ELEMENTS_DEFAULT"


class FlexivElementsError(RuntimeError):
    """Elements discovery is absent, ambiguous, or unsupported."""


@dataclass(frozen=True)
class FlexivElementsSelection:
    elements_series: str
    detected_version: str
    source: str
    source_path: Optional[str] = None

    def public_dict(self) -> dict[str, object]:
        return {
            "elements_series": self.elements_series,
            "elements_version": self.detected_version,
            "source": self.source,
            "source_path": self.source_path,
        }


@dataclass(frozen=True)
class _Candidate:
    version: str
    series: str
    source: str
    path: Path
    modified_ns: int


_ELEMENTS_VERSION_RE = re.compile(r"(?<!\d)(3\.(?:9|10|11)(?:\.\d+)*)(?!\d)")


def normalize_elements_version(value: object) -> tuple[str, str]:
    """Return ``(series, detected version)`` for a supported Elements value."""
    text = str(value or "").strip()
    match = _ELEMENTS_VERSION_RE.search(text)
    if match is None:
        raise FlexivElementsError(
            "Elements version must be in the supported 3.9, 3.10, or 3.11 series"
        )
    detected = match.group(1)
    series = ".".join(detected.split(".")[:2])
    return series, detected


def _safe_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except (FileNotFoundError, OSError, PermissionError):
        return 0


def _candidate(value: object, source: str, path: Path) -> Optional[_Candidate]:
    try:
        series, version = normalize_elements_version(value)
    except FlexivElementsError:
        return None
    return _Candidate(version, series, source, path, _safe_mtime_ns(path))


def _running_candidates(proc_root: Path) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    try:
        entries = list(proc_root.iterdir())
    except (FileNotFoundError, OSError, PermissionError):
        return candidates
    for entry in entries:
        if not entry.name.isdigit():
            continue
        command_path = entry / "cmdline"
        try:
            command = command_path.read_bytes()[: 1024 * 1024].replace(b"\0", b" ")
            text = command.decode("utf-8", errors="replace")
        except (FileNotFoundError, OSError, PermissionError):
            continue
        if "FlexivElementsStudio" not in text and "FSP_v" not in text:
            continue
        found = _candidate(text, "running-process", command_path)
        if found is not None:
            candidates.append(found)
    return candidates


def _log_prefix(path: Path, max_bytes: int = 256 * 1024) -> str:
    try:
        with path.open("rb") as stream:
            return stream.read(max_bytes).decode("utf-8", errors="replace")
    except (FileNotFoundError, OSError, PermissionError):
        return ""


def _log_candidates(home: Path) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for path in default_elements_cache_paths(home):
        if not path.name.startswith("UIApp_") or path.suffix != ".log":
            continue
        found = _candidate(str(path), "elements-log", path)
        if found is None:
            found = _candidate(_log_prefix(path), "elements-log", path)
        if found is not None:
            candidates.append(found)
    return sorted(candidates, key=lambda item: item.modified_ns, reverse=True)


def _installed_candidates(home: Path) -> list[_Candidate]:
    paths: list[Path] = []
    paths.extend(home.glob("FSP_v*"))
    paths.extend(home.glob("FSP_v*/FlexivElementsStudio_v*"))
    unique = {path.resolve(strict=False): path for path in paths}
    candidates = [
        found
        for path in unique.values()
        if (found := _candidate(str(path), "installed-elements", path)) is not None
    ]
    return sorted(candidates, key=lambda item: item.modified_ns, reverse=True)


def _selection_from_candidate(candidate: _Candidate) -> FlexivElementsSelection:
    return FlexivElementsSelection(
        elements_series=candidate.series,
        detected_version=candidate.version,
        source=candidate.source,
        source_path=str(candidate.path),
    )


def _unique_series(candidates: Iterable[_Candidate]) -> set[str]:
    return {candidate.series for candidate in candidates}


def select_flexiv_elements(
    explicit: Optional[str] = None,
    *,
    environment: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
    proc_root: Path = Path("/proc"),
) -> FlexivElementsSelection:
    """Detect Elements for display and diagnostics, never for RDK selection."""
    env = os.environ if environment is None else environment
    override = explicit or env.get(ELEMENTS_VERSION_ENV)
    if override:
        series, version = normalize_elements_version(override)
        return FlexivElementsSelection(
            series,
            version,
            "explicit" if explicit else "environment",
        )

    resolved_home = Path(home or env.get("HOME") or Path.home()).expanduser()
    running = _running_candidates(proc_root)
    running_series = _unique_series(running)
    if len(running_series) > 1:
        raise FlexivElementsError(
            "Multiple running Flexiv Elements series were detected: "
            + ", ".join(sorted(running_series))
        )
    if running:
        return _selection_from_candidate(max(running, key=lambda item: item.modified_ns))

    logs = _log_candidates(resolved_home)
    if logs:
        return _selection_from_candidate(logs[0])

    installed = _installed_candidates(resolved_home)
    installed_series = _unique_series(installed)
    if len(installed_series) == 1 and installed:
        return _selection_from_candidate(installed[0])
    if len(installed_series) > 1:
        raise FlexivElementsError(
            "Multiple Flexiv Elements series are installed but none is active"
        )

    fallback = env.get(ELEMENTS_DEFAULT_ENV)
    if fallback:
        series, version = normalize_elements_version(fallback)
        return FlexivElementsSelection(series, version, "environment-default")
    raise FlexivElementsError("Flexiv Elements version could not be detected")


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements-version")
    parser.add_argument("--home", type=Path)
    args = parser.parse_args(argv)
    selection = select_flexiv_elements(args.elements_version, home=args.home)
    print(json.dumps(selection.public_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "ELEMENTS_DEFAULT_ENV",
    "ELEMENTS_VERSION_ENV",
    "FlexivElementsError",
    "FlexivElementsSelection",
    "normalize_elements_version",
    "select_flexiv_elements",
]
