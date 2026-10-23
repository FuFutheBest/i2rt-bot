#!/usr/bin/env python3
"""Export accepted NEDF3 episodes in a time range as one LeRobot dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from incubator.control_panel.converter_profile import DEFAULT_PROFILE_ID, default_profile  # noqa: E402
from incubator.control_panel.recording_paths import iter_recording_episode_roots  # noqa: E402


MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_SELECTED_EPISODES = 1000
EPISODE_NAME = re.compile(r"^episode_[A-Za-z0-9_-]{1,120}$")
DATE_DIRECTORY = re.compile(r"^[0-9]{8}$")


class ExportError(RuntimeError):
    """The requested accepted-episode export cannot be completed safely."""


@dataclass(frozen=True)
class AcceptedEpisode:
    path: Path
    timestamp: datetime
    timestamp_source: str
    episode_id: Optional[str]
    episode_index: Optional[int]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def compute_job_id(
    source_fingerprint: str,
    profile_id: str = DEFAULT_PROFILE_ID,
) -> str:
    identity = {
        "source_fingerprint": source_fingerprint,
        "profile_id": profile_id,
    }
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ExportError(f"Cannot inspect JSON artifact: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise ExportError(f"JSON artifact is not a regular file: {path}")
    if info.st_size <= 0 or info.st_size > MAX_JSON_BYTES:
        raise ExportError(f"JSON artifact has invalid size: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExportError(f"JSON artifact is not an object: {path}")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _parse_timestamp(value: Any, local_timezone: ZoneInfo) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=local_timezone)
    return parsed.astimezone(local_timezone)


def _episode_timestamp(
    episode_path: Path,
    result: Mapping[str, Any],
    local_timezone: ZoneInfo,
) -> tuple[datetime, str]:
    parsed = _parse_timestamp(result.get("timestamp"), local_timezone)
    if parsed is not None:
        return parsed, "episode_result.timestamp"
    marker_path = episode_path / ".conversion_source.json"
    if marker_path.is_file() and not marker_path.is_symlink():
        try:
            marker = _read_json(marker_path)
        except ExportError:
            marker = {}
        parsed = _parse_timestamp(marker.get("finalized_at"), local_timezone)
        if parsed is not None:
            return parsed, ".conversion_source.finalized_at"
    try:
        modified = datetime.fromtimestamp(
            episode_path.stat().st_mtime, tz=local_timezone
        )
    except OSError as exc:
        raise ExportError(
            f"Cannot determine episode timestamp: {episode_path}"
        ) from exc
    return modified, "filesystem_mtime"


def _is_accepted(episode_path: Path, result: Mapping[str, Any]) -> bool:
    """Use applied whole-operation decisions for new recordings.

    Markers created before the decision contract remain compatible with their
    legacy episode_result classification.
    """
    decision_path = episode_path / ".recording_decision.json"
    if decision_path.exists():
        decision = _read_json(decision_path)
        return (
            decision.get("status") == "accepted"
            and decision.get("action") == "episode_accept"
        )
    marker_path = episode_path / ".conversion_source.json"
    if marker_path.exists():
        marker = _read_json(marker_path)
        if marker.get("decision_required") is True:
            return False
    return result.get("episode_result") == "success"


def discover_accepted_episodes(
    recording_root: Path,
    *,
    start: datetime,
    end: datetime,
    local_timezone: ZoneInfo,
) -> list[AcceptedEpisode]:
    root = Path(recording_root).expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ExportError(f"Recording root is not a real directory: {root}")
    selected: list[AcceptedEpisode] = []
    seen: set[Path] = set()
    for episode_root in iter_recording_episode_roots(root):
        if not episode_root.is_dir() or episode_root.is_symlink():
            continue
        for result_path in sorted(episode_root.glob("episode_*/episode_result.json")):
            episode_path = result_path.parent
            try:
                resolved = episode_path.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError):
                continue
            if resolved in seen or resolved.is_symlink() or not resolved.is_dir():
                continue
            seen.add(resolved)
            try:
                result = _read_json(resolved / "episode_result.json")
            except ExportError as exc:
                raise ExportError(
                    f"Cannot safely classify recorded episode {resolved}: {exc}"
                ) from exc
            if not _is_accepted(resolved, result):
                continue
            timestamp, timestamp_source = _episode_timestamp(
                resolved, result, local_timezone
            )
            if not start <= timestamp < end:
                continue
            raw_index = result.get("episode_index")
            episode_index = (
                int(raw_index)
                if isinstance(raw_index, int) and not isinstance(raw_index, bool)
                else None
            )
            episode_id = result.get("episode_id")
            selected.append(
                AcceptedEpisode(
                    path=resolved,
                    timestamp=timestamp,
                    timestamp_source=timestamp_source,
                    episode_id=episode_id if isinstance(episode_id, str) else None,
                    episode_index=episode_index,
                )
            )
    selected.sort(key=lambda item: (item.timestamp, str(item.path)))
    return selected


def _selected_episode_path(recording_root: Path, episode_id: str) -> Path:
    if not isinstance(episode_id, str) or not episode_id:
        raise ExportError("Selected episode id must be a non-empty string")
    parts = episode_id.split(":")
    if len(parts) == 1:
        date_name = None
        episode_name = parts[0]
    elif len(parts) == 2 and DATE_DIRECTORY.fullmatch(parts[0]):
        date_name, episode_name = parts
    else:
        raise ExportError(f"Invalid selected episode id: {episode_id}")
    if EPISODE_NAME.fullmatch(episode_name) is None:
        raise ExportError(f"Invalid selected episode id: {episode_id}")
    candidate = (
        recording_root / episode_name
        if date_name is None
        else recording_root / date_name / episode_name
    )
    try:
        info = candidate.lstat()
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(recording_root)
    except (OSError, ValueError) as exc:
        raise ExportError(f"Selected episode was not found: {episode_id}") from exc
    if candidate.is_symlink() or not candidate.is_dir() or not info:
        raise ExportError(f"Selected episode was not found: {episode_id}")
    return resolved


def discover_selected_accepted_episodes(
    recording_root: Path,
    episode_ids: Sequence[str],
    *,
    local_timezone: ZoneInfo,
) -> list[AcceptedEpisode]:
    """Resolve an explicit browser-safe selection without accepting paths."""

    root = Path(recording_root).expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ExportError(f"Recording root is not a real directory: {root}")
    requested = list(episode_ids)
    if not requested:
        raise ExportError("At least one episode must be selected")
    if len(requested) > MAX_SELECTED_EPISODES:
        raise ExportError(
            f"At most {MAX_SELECTED_EPISODES} episodes may be selected"
        )
    if len(set(requested)) != len(requested):
        raise ExportError("Selected episode ids must be unique")

    selected: list[AcceptedEpisode] = []
    for episode_id in requested:
        episode_path = _selected_episode_path(root, episode_id)
        try:
            result = _read_json(episode_path / "episode_result.json")
        except ExportError as exc:
            raise ExportError(
                f"Cannot safely classify selected episode {episode_id}: {exc}"
            ) from exc
        if not _is_accepted(episode_path, result):
            raise ExportError(
                f"Selected episode is not accepted: {episode_id}"
            )
        timestamp, timestamp_source = _episode_timestamp(
            episode_path, result, local_timezone
        )
        raw_index = result.get("episode_index")
        episode_index = (
            int(raw_index)
            if isinstance(raw_index, int) and not isinstance(raw_index, bool)
            else None
        )
        raw_episode_id = result.get("episode_id")
        selected.append(
            AcceptedEpisode(
                path=episode_path,
                timestamp=timestamp,
                timestamp_source=timestamp_source,
                episode_id=(
                    raw_episode_id if isinstance(raw_episode_id, str) else None
                ),
                episode_index=episode_index,
            )
        )
    selected.sort(key=lambda item: (item.timestamp, str(item.path)))
    return selected


def inspect_conversion(
    episode: AcceptedEpisode,
    *,
    conversion_root: Path,
    profile_id: str,
) -> dict[str, Any]:
    base = {
        "path": str(episode.path),
        "timestamp": episode.timestamp.isoformat(),
        "timestamp_source": episode.timestamp_source,
        "episode_id": episode.episode_id,
        "episode_index": episode.episode_index,
        "profile_id": profile_id,
    }
    marker_path = episode.path / ".conversion_source.json"
    if not marker_path.exists():
        return {
            **base,
            "state": "marker_missing",
            "error": f"Conversion source marker is missing: {marker_path}",
        }
    try:
        receipt = _read_json(marker_path)
    except ExportError as exc:
        return {**base, "state": "marker_invalid", "error": str(exc)}
    fingerprint = receipt.get("fingerprint")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("result") != "success"
        or not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        return {
            **base,
            "state": "marker_invalid",
            "error": "Invalid conversion source marker",
        }
    try:
        receipt_path = Path(str(receipt.get("path") or "")).resolve(strict=True)
    except (OSError, ValueError):
        return {
            **base,
            "state": "marker_invalid",
            "error": "Marker source path is unavailable",
        }
    if receipt_path != episode.path:
        return {
            **base,
            "state": "marker_invalid",
            "error": "Marker source path does not match episode",
        }
    if (
        episode.episode_id is not None
        and receipt.get("episode_id") != episode.episode_id
    ):
        return {
            **base,
            "state": "marker_invalid",
            "error": "Marker episode_id does not match result metadata",
        }
    if (
        episode.episode_index is not None
        and receipt.get("episode_index") != episode.episode_index
    ):
        return {
            **base,
            "state": "marker_invalid",
            "error": "Marker episode_index does not match result metadata",
        }

    job_id = compute_job_id(fingerprint, profile_id)
    identity = {
        "source_fingerprint": fingerprint,
        "profile_id": profile_id,
    }
    enriched = {
        **base,
        "fingerprint": fingerprint,
        "job_id": job_id,
        "dense_labels_sha256": receipt.get("dense_labels_sha256"),
    }
    job_path = conversion_root / "jobs" / f"{job_id}.json"
    if not job_path.exists():
        return {
            **enriched,
            "state": "job_missing",
            "error": f"Conversion job is missing: {job_path}",
        }
    try:
        job = _read_json(job_path)
    except ExportError as exc:
        return {**enriched, "state": "job_invalid", "error": str(exc)}
    if job.get("job_id") != job_id or job.get("identity") != identity:
        return {
            **enriched,
            "state": "job_invalid",
            "error": "Conversion job identity mismatch",
        }
    job_source = job.get("source")
    if not isinstance(job_source, Mapping):
        return {
            **enriched,
            "state": "job_invalid",
            "error": "Conversion job has no source identity",
        }
    try:
        job_source_path = Path(str(job_source.get("path") or "")).resolve(strict=True)
    except (OSError, ValueError):
        return {
            **enriched,
            "state": "job_invalid",
            "error": "Conversion job source path is unavailable",
        }
    if job_source_path != episode.path or job_source.get("fingerprint") != fingerprint:
        return {
            **enriched,
            "state": "job_invalid",
            "error": "Conversion job source does not match episode",
        }
    state = str(job.get("state") or "unknown")
    if state != "ready":
        return {
            **enriched,
            "state": state,
            "error": job.get("error"),
        }

    shard = conversion_root / "published" / job_id
    if not shard.exists():
        return {
            **enriched,
            "state": "shard_missing",
            "error": f"Published shard is missing: {shard}",
        }
    try:
        shard = shard.resolve(strict=True)
        shard.relative_to((conversion_root / "published").resolve(strict=True))
    except (OSError, ValueError) as exc:
        return {**enriched, "state": "shard_missing", "error": str(exc)}
    try:
        manifest = _read_json(shard / "meta" / "shard_manifest.json")
    except ExportError as exc:
        return {**enriched, "state": "shard_invalid", "error": str(exc)}
    if (
        manifest.get("job_id") != job_id
        or manifest.get("profile_id") != profile_id
        or manifest.get("source_fingerprint") != fingerprint
    ):
        return {
            **enriched,
            "state": "shard_invalid",
            "error": "Published shard identity mismatch",
        }
    return {**enriched, "state": "ready", "shard_path": str(shard), "error": None}


def resolve_conversions(
    episodes: Sequence[AcceptedEpisode],
    *,
    conversion_root: Path,
    profile_id: str,
    wait_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> list[dict[str, Any]]:
    root = Path(conversion_root).expanduser().resolve(strict=False)
    deadline = monotonic() + max(0.0, float(wait_seconds))
    while True:
        results = [
            inspect_conversion(
                episode,
                conversion_root=root,
                profile_id=profile_id,
            )
            for episode in episodes
        ]
        if all(item["state"] == "ready" for item in results):
            return results
        remaining = deadline - monotonic()
        if remaining <= 0:
            return results
        sleep(min(1.0, remaining))


def _selection_manifest(
    *,
    recording_root: Path,
    conversion_root: Path,
    start: datetime,
    end: datetime,
    timezone_name: str,
    profile_id: str,
    dataset_name: str,
    conversions: Sequence[Mapping[str, Any]],
    selection: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "dataset_name": dataset_name,
        "selection": dict(
            selection
            or {
                "mode": "time_range",
                "start_inclusive": start.isoformat(),
                "end_exclusive": end.isoformat(),
                "timezone": timezone_name,
                "episode_result": "success",
            }
        ),
        "recording_root": str(recording_root),
        "conversion_root": str(conversion_root),
        "profile_id": profile_id,
        "episode_count": len(conversions),
        "episodes": [dict(item) for item in conversions],
    }


def export_accepted_range(
    episodes: Sequence[AcceptedEpisode],
    conversions: Sequence[Mapping[str, Any]],
    *,
    recording_root: Path,
    conversion_root: Path,
    output_dir: Path,
    dataset_name: str,
    start: datetime,
    end: datetime,
    timezone_name: str,
    profile_id: str,
    link_mode: str,
    merge_fn: Optional[Callable[..., Path]] = None,
    selection: Optional[Mapping[str, Any]] = None,
) -> Path:
    if (
        not dataset_name
        or Path(dataset_name).name != dataset_name
        or dataset_name in {".", ".."}
    ):
        raise ExportError("dataset_name must be one safe path component")
    if not episodes:
        raise ExportError("No accepted episodes were found in the requested range")
    if len(episodes) != len(conversions):
        raise ExportError(
            "Conversion inspection count does not match episode selection"
        )
    blocked = [item for item in conversions if item.get("state") != "ready"]
    if blocked:
        states: dict[str, int] = {}
        for item in blocked:
            state = str(item.get("state") or "unknown")
            states[state] = states.get(state, 0) + 1
        detail = ", ".join(
            f"{state}={count}" for state, count in sorted(states.items())
        )
        raise ExportError(f"Selected episodes are not all conversion-ready: {detail}")
    shard_paths = [Path(str(item["shard_path"])) for item in conversions]
    if len(set(shard_paths)) != len(shard_paths):
        raise ExportError("Multiple accepted episodes resolved to the same shard")
    if merge_fn is None:
        from incubator.control_panel.tools.merge_lerobot_shards import merge_shards

        merge_fn = merge_shards
    dataset_root = merge_fn(
        shard_paths,
        Path(output_dir),
        dataset_name,
        link_mode=link_mode,
    )
    manifest = _selection_manifest(
        recording_root=Path(recording_root),
        conversion_root=Path(conversion_root),
        start=start,
        end=end,
        timezone_name=timezone_name,
        profile_id=profile_id,
        dataset_name=dataset_name,
        conversions=conversions,
        selection=selection,
    )
    _atomic_write_json(Path(dataset_root) / "meta" / "export_selection.json", manifest)
    return Path(dataset_root)


def resolve_time_range(
    *,
    date_text: Optional[str],
    start_text: Optional[str],
    end_text: Optional[str],
    timezone_name: str,
) -> tuple[datetime, datetime, ZoneInfo]:
    try:
        local_timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ExportError(f"Unknown timezone: {timezone_name}") from exc
    if date_text and (start_text or end_text):
        raise ExportError("Use either --date or --start/--end, not both")
    if bool(start_text) != bool(end_text):
        raise ExportError("--start and --end must be provided together")
    if date_text:
        try:
            selected_date = date.fromisoformat(date_text)
        except ValueError as exc:
            raise ExportError("--date must use YYYY-MM-DD") from exc
        start = datetime.combine(selected_date, datetime_time.min, local_timezone)
        return start, start + timedelta(days=1), local_timezone
    if start_text and end_text:
        start = _parse_timestamp(start_text, local_timezone)
        end = _parse_timestamp(end_text, local_timezone)
        if start is None or end is None:
            raise ExportError("--start and --end must be ISO-8601 timestamps")
        if end <= start:
            raise ExportError("--end must be later than --start")
        return start, end, local_timezone
    today = datetime.now(local_timezone).date()
    start = datetime.combine(today, datetime_time.min, local_timezone)
    return start, start + timedelta(days=1), local_timezone


def _default_dataset_name(start: datetime, end: datetime) -> str:
    if end - start == timedelta(days=1) and start.time() == datetime_time.min:
        return f"accepted-{start:%Y%m%d}"
    return f"accepted-{start:%Y%m%dT%H%M%S}-{end:%Y%m%dT%H%M%S}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select whole accepted NEDF3 episodes by local time and merge their "
            "ready LeRobot shards without video re-encoding."
        )
    )
    parser.add_argument("--recording-root", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--date", dest="date_text")
    parser.add_argument("--start", dest="start_text")
    parser.add_argument("--end", dest="end_text")
    parser.add_argument(
        "--episode-id",
        dest="episode_ids",
        action="append",
        default=[],
        help=(
            "Select one public recording id (repeatable), for example "
            "20260726:episode_0001. Cannot be combined with time filters."
        ),
    )
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--wait-seconds", type=float, default=0.0)
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hardlink", "reflink", "copy"),
        default="auto",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        recording_root = args.recording_root.expanduser().resolve(strict=True)
        explicit_selection = bool(args.episode_ids)
        if explicit_selection and (
            args.date_text or args.start_text or args.end_text
        ):
            raise ExportError(
                "--episode-id cannot be combined with --date or --start/--end"
            )
        if explicit_selection:
            try:
                local_timezone = ZoneInfo(args.timezone)
            except ZoneInfoNotFoundError as exc:
                raise ExportError(f"Unknown timezone: {args.timezone}") from exc
            episodes = discover_selected_accepted_episodes(
                recording_root,
                args.episode_ids,
                local_timezone=local_timezone,
            )
            start = min(item.timestamp for item in episodes)
            end = max(item.timestamp for item in episodes) + timedelta(
                microseconds=1
            )
            selection = {
                "mode": "episode_ids",
                "episode_ids": list(args.episode_ids),
                "timezone": args.timezone,
                "episode_result": "success",
            }
        else:
            start, end, local_timezone = resolve_time_range(
                date_text=args.date_text,
                start_text=args.start_text,
                end_text=args.end_text,
                timezone_name=args.timezone,
            )
            episodes = discover_accepted_episodes(
                recording_root,
                start=start,
                end=end,
                local_timezone=local_timezone,
            )
            selection = {
                "mode": "time_range",
                "start_inclusive": start.isoformat(),
                "end_exclusive": end.isoformat(),
                "timezone": args.timezone,
                "episode_result": "success",
            }
        profile = default_profile()
        conversion_root = (
            Path(
                args.conversion_root
                if args.conversion_root is not None
                else profile.output_root
            )
            .expanduser()
            .resolve(strict=False)
        )
        output_dir = (
            Path(
                args.output_dir
                if args.output_dir is not None
                else recording_root.parent / "lerobot_exports"
            )
            .expanduser()
            .resolve(strict=False)
        )
        dataset_name = args.dataset_name or (
            f"selected-{datetime.now(local_timezone):%Y%m%d-%H%M%S}"
            if explicit_selection
            else _default_dataset_name(start, end)
        )
        conversions = resolve_conversions(
            episodes,
            conversion_root=conversion_root,
            profile_id=profile.profile_id,
            wait_seconds=max(0.0, args.wait_seconds),
        )
        report: dict[str, Any] = {
            "ok": True,
            "dry_run": bool(args.dry_run),
            "selection": selection,
            "recording_root": str(recording_root),
            "conversion_root": str(conversion_root),
            "output_dir": str(output_dir),
            "dataset_name": dataset_name,
            "episode_count": len(episodes),
            "ready_count": sum(item["state"] == "ready" for item in conversions),
            "episodes": conversions,
        }
        if not args.dry_run:
            dataset_root = export_accepted_range(
                episodes,
                conversions,
                recording_root=recording_root,
                conversion_root=conversion_root,
                output_dir=output_dir,
                dataset_name=dataset_name,
                start=start,
                end=end,
                timezone_name=args.timezone,
                profile_id=profile.profile_id,
                link_mode=args.link_mode,
                selection=selection,
            )
            report["dataset_root"] = str(dataset_root)
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except (ExportError, FileExistsError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"ok": False, "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
