"""Read-only duration statistics for finalized NEDF3 episodes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import time
from typing import Any

from utils.episode_time_ledger import (
    LEDGER_FILENAME,
    append_timing_records,
    read_episode_time_range_ns,
    read_timing_ledger,
)
from utils.logger_config import logger


UNLEDGERED_EPISODE_GRACE_SECONDS = 60.0


@dataclass(frozen=True)
class DailyEpisodeStats:
    episode_count: int = 0
    duration_seconds: float = 0.0
    total_episode_count: int = 0
    total_duration_seconds: float = 0.0
    anomaly_error_count: int = 0
    manual_discard_count: int = 0
    unreadable_count: int = 0


def _iter_finalized_episode_dirs(output_dir: Path) -> Iterable[Path]:
    if not output_dir.is_dir():
        return
    candidates = list(output_dir.iterdir())
    for child in list(candidates):
        if child.is_dir() and not child.name.startswith("episode_"):
            candidates.extend(child.iterdir())
    for child in candidates:
        if (
            not child.is_dir()
            or not child.name.startswith("episode_")
            or child.name.endswith("_temp")
        ):
            continue
        chunk_dir = child / "data"
        has_data = (child / "data.mcap").is_file() or (
            (chunk_dir / "mcap_index.json").is_file()
            and any(chunk_dir.glob("*.mcap"))
        )
        if (child / "metadata.json").is_file() and has_data:
            yield child


def _latest_data_mtime(episode_dir: Path) -> float:
    """Return the newest recording-file mtime, excluding later upload state."""
    data_paths = [episode_dir / "metadata.json", episode_dir / "data.mcap"]
    data_dir = episode_dir / "data"
    if data_dir.is_dir():
        data_paths.extend(path for path in data_dir.rglob("*") if path.is_file())
    latest = 0.0
    for path in data_paths:
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            continue
    return latest


def _read_latest_ledger_records(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    latest_rank: dict[tuple[str, str], tuple[int, float, int]] = {}
    sequence = 0
    ledger_paths = [root / LEDGER_FILENAME]
    ledger_paths.extend(
        child / LEDGER_FILENAME for child in root.iterdir() if child.is_dir()
    )
    for ledger_path in ledger_paths:
        if not ledger_path.is_file():
            continue
        for record in read_timing_ledger(ledger_path):
            sequence += 1
            episode_name = str(record.get("episode_name") or "")
            if episode_name:
                task_key = str(
                    Path(record.get("task_output_dir") or ledger_path.parent).resolve()
                )
                key = (task_key, episode_name)
                try:
                    completed_at = datetime.fromisoformat(
                        str(record.get("completed_at") or "").replace("Z", "+00:00")
                    ).timestamp()
                except (TypeError, ValueError, OverflowError):
                    completed_at = float("-inf")
                rank = (
                    0 if bool(record.get("backfilled")) else 1,
                    completed_at,
                    sequence,
                )
                if rank > latest_rank.get(key, (-1, float("-inf"), -1)):
                    latest[key] = record
                    latest_rank[key] = rank
    return latest


def scan_today_episode_stats(
    output_dir: str,
    *,
    today: date | None = None,
    backfill_missing: bool = False,
) -> DailyEpisodeStats:
    """Sum timestamp ranges for episodes whose recording files were saved today."""
    local_timezone = datetime.now().astimezone().tzinfo
    target_date = today or datetime.now(tz=local_timezone).date()
    root = Path(output_dir).expanduser()
    effective_duration_seconds = 0.0
    effective_episode_count = 0
    total_duration_seconds = 0.0
    total_episode_count = 0
    anomaly_error_count = 0
    manual_discard_count = 0
    unreadable_count = 0
    backfill_records: list[dict[str, Any]] = []

    ledger_records = _read_latest_ledger_records(root) if root.is_dir() else {}
    ledger_keys: set[tuple[str, str]] = set()
    for key, record in ledger_records.items():
        try:
            completed_at = datetime.fromisoformat(
                str(record.get("completed_at") or "").replace("Z", "+00:00")
            )
            if completed_at.astimezone(local_timezone).date() != target_date:
                continue
            status = str(record.get("status") or "")
            if status not in {"accepted", "anomaly_error", "manual_discard"}:
                continue
            ledger_keys.add(key)
            duration = record.get("duration_seconds")
            if status == "manual_discard":
                manual_discard_count += 1
            if duration is None:
                if status in {"anomaly_error", "manual_discard"}:
                    unreadable_count += 1
                continue
            duration_seconds = max(0.0, float(duration))
            total_duration_seconds += duration_seconds
            total_episode_count += 1
            if status == "accepted":
                effective_duration_seconds += duration_seconds
                effective_episode_count += 1
            elif status == "anomaly_error":
                anomaly_error_count += 1
        except (TypeError, ValueError, OverflowError):
            unreadable_count += 1

    for episode_dir in _iter_finalized_episode_dirs(root):
        episode_key = (str(episode_dir.parent.resolve()), episode_dir.name)
        if episode_key in ledger_keys:
            continue
        try:
            # Finalized episodes are immutable. The mtime prefilter prevents old
            # collections from being opened on every GUI startup.
            data_mtime = _latest_data_mtime(episode_dir)
            completed_at = datetime.fromtimestamp(data_mtime, tz=local_timezone)
            if completed_at.date() != target_date:
                continue
            if time.time() - data_mtime < UNLEDGERED_EPISODE_GRACE_SECONDS:
                continue
            bounds = read_episode_time_range_ns(str(episode_dir.resolve()))
            if bounds is None:
                unreadable_count += 1
                continue
            first_timestamp_ns, last_timestamp_ns = bounds
            duration_ns = last_timestamp_ns - first_timestamp_ns
            if duration_ns < 0:
                unreadable_count += 1
                continue
            duration_seconds = duration_ns / 1e9
            effective_duration_seconds += duration_seconds
            total_duration_seconds += duration_seconds
            effective_episode_count += 1
            total_episode_count += 1
            if backfill_missing:
                backfill_records.append(
                    {
                        "version": 1,
                        "episode_name": episode_dir.name,
                        "task_name": episode_dir.parent.name,
                        "task_output_dir": str(episode_dir.parent.resolve()),
                        "status": "accepted",
                        "first_timestamp_ns": first_timestamp_ns,
                        "last_timestamp_ns": last_timestamp_ns,
                        "duration_seconds": duration_seconds,
                        "completed_at": completed_at.isoformat(),
                        "backfilled": True,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - malformed SDK episodes are skipped
            unreadable_count += 1
            logger.warning(f"Failed to read episode duration from {episode_dir}: {exc}")

    if backfill_records:
        try:
            append_timing_records(root / LEDGER_FILENAME, backfill_records)
            logger.info(f"Backfilled timing ledger for {len(backfill_records)} finalized episode(s)")
        except OSError as exc:
            logger.warning(f"Failed to backfill episode timing ledger: {exc}")

    return DailyEpisodeStats(
        episode_count=effective_episode_count,
        duration_seconds=effective_duration_seconds,
        total_episode_count=total_episode_count,
        total_duration_seconds=total_duration_seconds,
        anomaly_error_count=anomaly_error_count,
        manual_discard_count=manual_discard_count,
        unreadable_count=unreadable_count,
    )


def format_duration(total_seconds: float) -> str:
    """Format a cumulative duration without wrapping after 24 hours."""
    seconds = max(0, round(total_seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

