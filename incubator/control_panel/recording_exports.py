"""Durable, low-priority batch export of selected recording episodes."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
from typing import Any, Callable, Mapping, Optional, Sequence
import uuid
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .conversion_jobs import ConversionJobManager
from .tools.export_accepted_lerobot_range import (
    AcceptedEpisode,
    ExportError,
    MAX_SELECTED_EPISODES,
    discover_selected_accepted_episodes,
    inspect_conversion,
)


SCHEMA_VERSION = 1
VALID_STATES = frozenset({"queued", "waiting", "running", "ready", "error"})
TRANSIENT_CONVERSION_STATES = frozenset({"queued", "running", "retry"})
DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MAX_JOB_BYTES = 4 * 1024 * 1024
DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_EXPORT_SCRIPT = (
    Path(__file__).resolve().parent / "tools" / "export_accepted_lerobot_range.py"
)
DEFAULT_QUEUE_ROOT = (
    Path.home()
    / ".local"
    / "share"
    / "superinference-control-panel"
    / "recording_exports"
)


class RecordingExportError(RuntimeError):
    """A selected recording export cannot be queued or completed safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise RecordingExportError(f"Export queue is not a real directory: {path}")
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _ensure_private_directory(path.parent)
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_JOB_BYTES:
        raise RecordingExportError("Export job exceeds size limit")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        try:
            os.fchmod(descriptor, 0o600)
        except (AttributeError, OSError):
            pass
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RecordingExportError(f"Export job is unavailable: {path.name}") from exc
    if path.is_symlink() or not path.is_file() or not 0 < info.st_size <= MAX_JOB_BYTES:
        raise RecordingExportError(f"Unsafe export job: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecordingExportError(f"Unreadable export job: {path.name}") from exc
    if not isinstance(value, dict):
        raise RecordingExportError(f"Export job is not an object: {path.name}")
    return value


def _state_counts(values: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        state = str(value.get("state") or "unknown")
        counts[state] = counts.get(state, 0) + 1
    return counts


class RecordingExportManager:
    """One-worker persistent scheduler for explicit episode selections."""

    def __init__(
        self,
        recording_root_provider: Callable[[], Optional[Path]],
        conversion_jobs: ConversionJobManager,
        *,
        queue_root: Optional[Path] = None,
        output_root: Optional[Path] = None,
        export_python: Optional[Path] = None,
        export_script: Path = DEFAULT_EXPORT_SCRIPT,
        timezone_name: str = DEFAULT_TIMEZONE,
        busy: Callable[[], bool] = lambda: False,
        process_runner: Callable[..., Any] = subprocess.run,
        poll_interval_s: float = 1.0,
        process_timeout_s: float = 3600.0,
        auto_start: bool = True,
    ) -> None:
        configured_queue = os.environ.get(
            "SUPERINFERENCE_RECORDING_EXPORT_QUEUE_ROOT"
        )
        data_root = os.environ.get("SUPERINFERENCE_DATA_ROOT")
        selected_queue = queue_root
        if selected_queue is None and configured_queue:
            selected_queue = Path(configured_queue)
        if selected_queue is None and data_root:
            selected_queue = Path(data_root) / "recording_exports"
        self.queue_root = Path(selected_queue or DEFAULT_QUEUE_ROOT).expanduser().resolve(
            strict=False
        )
        self.jobs_dir = self.queue_root / "jobs"
        _ensure_private_directory(self.jobs_dir)

        configured_output = os.environ.get("SUPERINFERENCE_LEROBOT_EXPORT_ROOT")
        self._configured_output_root = (
            Path(output_root or configured_output).expanduser().resolve(strict=False)
            if output_root is not None or configured_output
            else None
        )
        self._recording_root_provider = recording_root_provider
        self.conversion_root = Path(
            getattr(
                conversion_jobs,
                "queue_root",
                self.queue_root.parent / "conversion_jobs",
            )
        ).expanduser().resolve(strict=False)
        self.profile_id = str(
            getattr(conversion_jobs, "profile_id", "unavailable-profile")
        )
        self.export_python = Path(
            export_python
            or getattr(conversion_jobs, "converter_python", Path(sys.executable))
        ).expanduser().resolve(strict=False)
        self.export_script = Path(export_script).expanduser().resolve(strict=False)
        try:
            self.local_timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise RecordingExportError(f"Unknown export timezone: {timezone_name}") from exc
        self.timezone_name = timezone_name
        self._busy = busy
        self._process_runner = process_runner
        self.poll_interval_s = max(0.1, min(float(poll_interval_s), 10.0))
        self.process_timeout_s = max(1.0, float(process_timeout_s))
        self.auto_start = bool(auto_start)
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._recover_interrupted_jobs()

    def _job_path(self, job_id: str) -> Path:
        if len(job_id) != 32 or any(character not in "0123456789abcdef" for character in job_id):
            raise RecordingExportError("Unsafe export job id")
        return self.jobs_dir / f"{job_id}.json"

    def _ordered_job_paths(self, *, newest_first: bool = False) -> list[Path]:
        ordered: list[tuple[int, str, Path]] = []
        for path in self.jobs_dir.glob("*.json"):
            try:
                ordered.append((path.stat().st_mtime_ns, path.name, path))
            except OSError:
                continue
        ordered.sort(reverse=newest_first)
        return [item[2] for item in ordered]

    def _recording_root(self) -> Path:
        raw_root = self._recording_root_provider()
        if raw_root is None:
            raise RecordingExportError("Recording root is not configured")
        try:
            root = Path(raw_root).expanduser().resolve(strict=True)
        except OSError as exc:
            raise RecordingExportError("Recording root is not available") from exc
        if root.is_symlink() or not root.is_dir():
            raise RecordingExportError("Recording root is not a real directory")
        return root

    def _output_root(self, recording_root: Path) -> Path:
        root = self._configured_output_root or recording_root.parent / "lerobot_exports"
        root = Path(root).expanduser().resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise RecordingExportError("LeRobot export root is not a real directory")
        return root

    @staticmethod
    def _dataset_name(value: Any, timezone_value: ZoneInfo) -> str:
        if value is None or not str(value).strip():
            return (
                f"selected-{datetime.now(timezone_value):%Y%m%d-%H%M%S}-"
                f"{uuid.uuid4().hex[:6]}"
            )
        name = str(value).strip()
        if DATASET_NAME.fullmatch(name) is None or name in {".", ".."}:
            raise RecordingExportError(
                "dataset_name must be 1-128 ASCII letters, digits, '.', '_' or '-'"
            )
        return name

    @staticmethod
    def _episode_record(episode: AcceptedEpisode) -> dict[str, Any]:
        return {
            "path": str(episode.path),
            "timestamp": episode.timestamp.isoformat(),
            "timestamp_source": episode.timestamp_source,
            "episode_id": episode.episode_id,
            "episode_index": episode.episode_index,
        }

    @staticmethod
    def _accepted_episode(value: Mapping[str, Any]) -> AcceptedEpisode:
        try:
            timestamp = datetime.fromisoformat(str(value["timestamp"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RecordingExportError("Export job has invalid episode metadata") from exc
        if timestamp.tzinfo is None:
            raise RecordingExportError("Export job episode timestamp has no timezone")
        path = Path(str(value.get("path") or "")).expanduser().resolve(strict=False)
        return AcceptedEpisode(
            path=path,
            timestamp=timestamp,
            timestamp_source=str(value.get("timestamp_source") or "unknown"),
            episode_id=(
                str(value["episode_id"])
                if isinstance(value.get("episode_id"), str)
                else None
            ),
            episode_index=(
                int(value["episode_index"])
                if isinstance(value.get("episode_index"), int)
                and not isinstance(value.get("episode_index"), bool)
                else None
            ),
        )

    def _conversion_status(self, job: Mapping[str, Any]) -> list[dict[str, Any]]:
        source = job.get("source")
        raw_episodes = source.get("episodes") if isinstance(source, Mapping) else None
        if not isinstance(raw_episodes, list):
            raise RecordingExportError("Export job has no episode selection")
        episodes = [
            self._accepted_episode(value)
            for value in raw_episodes
            if isinstance(value, Mapping)
        ]
        if len(episodes) != len(raw_episodes):
            raise RecordingExportError("Export job has invalid episode selection")
        return [
            inspect_conversion(
                episode,
                conversion_root=self.conversion_root,
                profile_id=self.profile_id,
            )
            for episode in episodes
        ]

    def _assert_dataset_available(self, output_root: Path, dataset_name: str) -> None:
        target = output_root / dataset_name
        if target.exists():
            raise RecordingExportError(f"Output dataset already exists: {target}")
        for path in self._ordered_job_paths():
            try:
                existing = _read_json(path)
            except RecordingExportError:
                continue
            if existing.get("state") in {"ready", "error"}:
                continue
            output = existing.get("output")
            if not isinstance(output, Mapping):
                continue
            if (
                output.get("dataset_name") == dataset_name
                and output.get("root") == str(output_root)
            ):
                raise RecordingExportError(
                    f"An export already owns dataset name: {dataset_name}"
                )

    def enqueue(
        self,
        episode_ids: Sequence[str],
        *,
        dataset_name: Any = None,
    ) -> dict[str, Any]:
        if isinstance(episode_ids, (str, bytes)) or not isinstance(
            episode_ids, Sequence
        ):
            raise RecordingExportError("episode_ids must be an array")
        requested = list(episode_ids)
        if len(requested) > MAX_SELECTED_EPISODES:
            raise RecordingExportError(
                f"At most {MAX_SELECTED_EPISODES} episodes may be selected"
            )
        recording_root = self._recording_root()
        try:
            episodes = discover_selected_accepted_episodes(
                recording_root,
                requested,
                local_timezone=self.local_timezone,
            )
        except (ExportError, OSError, ValueError) as exc:
            raise RecordingExportError(str(exc)) from exc
        selected_name = self._dataset_name(dataset_name, self.local_timezone)
        output_root = self._output_root(recording_root)
        with self._lock:
            self._assert_dataset_available(output_root, selected_name)
            job_id = uuid.uuid4().hex
            now = _utc_now()
            job: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "job_id": job_id,
                "state": "queued",
                "phase": "checking_conversion",
                "source": {
                    "recording_root": str(recording_root),
                    "episode_ids": requested,
                    "episode_count": len(episodes),
                    "episodes": [self._episode_record(item) for item in episodes],
                },
                "conversion": {
                    "root": str(self.conversion_root),
                    "profile_id": self.profile_id,
                    "ready_count": 0,
                    "states": {},
                },
                "output": {
                    "root": str(output_root),
                    "dataset_name": selected_name,
                    "dataset_root": str(output_root / selected_name),
                },
                "error": None,
                "created_at": now,
                "updated_at": now,
                "finished_at": None,
            }
            conversions = self._conversion_status(job)
            self._apply_readiness(job, conversions)
            _atomic_write_json(self._job_path(job_id), job)
        if self.auto_start:
            self.start()
            self._wake.set()
        return self._public_job(job)

    @staticmethod
    def _apply_readiness(
        job: dict[str, Any], conversions: Sequence[Mapping[str, Any]]
    ) -> bool:
        counts = _state_counts(conversions)
        ready_count = counts.get("ready", 0)
        conversion = job.setdefault("conversion", {})
        changed = (
            conversion.get("ready_count") != ready_count
            or conversion.get("states") != counts
        )
        conversion["ready_count"] = ready_count
        conversion["states"] = counts
        return changed

    @staticmethod
    def _blocked_conversion_states(
        conversions: Sequence[Mapping[str, Any]],
    ) -> dict[str, int]:
        blocked = [
            value
            for value in conversions
            if value.get("state") != "ready"
            and value.get("state") not in TRANSIENT_CONVERSION_STATES
        ]
        return _state_counts(blocked)

    def _refresh_job(self, path: Path, job: dict[str, Any]) -> bool:
        conversions = self._conversion_status(job)
        readiness_changed = self._apply_readiness(job, conversions)
        blocked = self._blocked_conversion_states(conversions)
        episode_count = int(job.get("source", {}).get("episode_count") or 0)
        ready_count = int(job.get("conversion", {}).get("ready_count") or 0)
        previous = (job.get("state"), job.get("phase"), job.get("error"))
        if blocked:
            detail = ", ".join(
                f"{state}={count}" for state, count in sorted(blocked.items())
            )
            job["state"] = "error"
            job["phase"] = "conversion_blocked"
            job["error"] = (
                "Selected episodes are not all conversion-ready: " + detail
            )
            job["finished_at"] = _utc_now()
        elif ready_count < episode_count:
            job["state"] = "waiting"
            job["phase"] = "waiting_conversion"
            job["error"] = None
        elif self._busy():
            job["state"] = "waiting"
            job["phase"] = "waiting_idle"
            job["error"] = None
        else:
            job["state"] = "queued"
            job["phase"] = "ready_to_merge"
            job["error"] = None
        changed = readiness_changed or previous != (
            job.get("state"),
            job.get("phase"),
            job.get("error"),
        )
        if changed:
            job["updated_at"] = _utc_now()
            _atomic_write_json(path, job)
        return job.get("phase") == "ready_to_merge"

    def _eligible_job(self) -> Optional[tuple[Path, dict[str, Any]]]:
        for path in self._ordered_job_paths():
            try:
                job = _read_json(path)
            except RecordingExportError:
                continue
            if job.get("state") not in {"queued", "waiting"}:
                continue
            try:
                if self._refresh_job(path, job):
                    return path, job
            except Exception as exc:
                job["state"] = "error"
                job["phase"] = "validation_failed"
                job["error"] = f"{type(exc).__name__}: {exc}"[:1200]
                job["updated_at"] = _utc_now()
                job["finished_at"] = job["updated_at"]
                _atomic_write_json(path, job)
        return None

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RecordingExportError("Recording export manager is closed")
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="recording-exports",
                daemon=True,
            )
            self._thread.start()

    def close(self, timeout_s: float = 3.0) -> None:
        with self._lock:
            self._closed = True
            self._stop.set()
            self._wake.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, timeout_s))
            if thread.is_alive():
                raise RecordingExportError(
                    "Recording export worker is still active after shutdown timeout"
                )
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                selected = self._eligible_job()
                if selected is not None:
                    self._execute(*selected)
                    continue
            except Exception:
                pass
            self._wake.wait(self.poll_interval_s)
            self._wake.clear()

    def _command(self, job: Mapping[str, Any]) -> list[str]:
        source = job["source"]
        output = job["output"]
        base = [
            str(self.export_python),
            str(self.export_script),
            "--recording-root",
            str(source["recording_root"]),
            "--conversion-root",
            str(self.conversion_root),
            "--output-dir",
            str(output["root"]),
            "--dataset-name",
            str(output["dataset_name"]),
            "--timezone",
            self.timezone_name,
            "--link-mode",
            "auto",
        ]
        for episode_id in source["episode_ids"]:
            base.extend(("--episode-id", str(episode_id)))
        wrappers = ["/usr/bin/systemd-run", "/usr/bin/ionice", "/usr/bin/nice"]
        if os.name == "posix" and all(Path(path).is_file() for path in wrappers):
            return [
                wrappers[0],
                "--user",
                "--scope",
                "--quiet",
                "-p",
                "CPUQuota=200%",
                "-p",
                "MemoryMax=8G",
                "-p",
                "TasksMax=128",
                wrappers[1],
                "-c",
                "3",
                wrappers[2],
                "-n",
                "19",
                *base,
            ]
        return base

    @staticmethod
    def _validate_output(job: Mapping[str, Any]) -> dict[str, Any]:
        output = job.get("output")
        if not isinstance(output, Mapping):
            raise RecordingExportError("Export job has no output contract")
        root = Path(str(output.get("dataset_root") or "")).resolve(strict=True)
        expected = (
            Path(str(output.get("root") or "")).resolve(strict=True)
            / str(output.get("dataset_name") or "")
        )
        if root != expected or root.is_symlink() or not root.is_dir():
            raise RecordingExportError("Published export path does not match the job")
        required = (
            "meta/info.json",
            "meta/episodes.jsonl",
            "meta/export_selection.json",
            "meta/merge_manifest.json",
            "meta/validation_report.json",
        )
        for relative in required:
            path = root / relative
            if path.is_symlink() or not path.is_file():
                raise RecordingExportError(
                    f"Published export is missing {relative}"
                )
        selection = _read_json(root / "meta" / "export_selection.json")
        if selection.get("episode_count") != job.get("source", {}).get(
            "episode_count"
        ):
            raise RecordingExportError("Published export selection count mismatch")
        merge_manifest = _read_json(root / "meta" / "merge_manifest.json")
        validation = _read_json(root / "meta" / "validation_report.json")
        if validation.get("status") != "passed":
            raise RecordingExportError("Published export validation did not pass")
        return {
            "dataset_root": str(root),
            "video_materialization": dict(
                merge_manifest.get("video_materialization") or {}
            ),
            "validation": validation,
        }

    def _execute(self, path: Path, job: dict[str, Any]) -> None:
        with self._lock:
            current = _read_json(path)
            if current.get("state") not in {"queued", "waiting"}:
                return
            current["state"] = "running"
            current["phase"] = "merging"
            current["error"] = None
            current["updated_at"] = _utc_now()
            _atomic_write_json(path, current)
            job = current

        try:
            completed = self._process_runner(
                self._command(job),
                cwd=str(self.export_script.parents[2]),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                start_new_session=True,
                timeout=self.process_timeout_s,
            )
            if int(completed.returncode) != 0:
                detail = (
                    completed.stderr or completed.stdout or "LeRobot export failed"
                ).strip()[-1600:]
                raise RecordingExportError(detail)
            job["result"] = self._validate_output(job)
            job["state"] = "ready"
            job["phase"] = "published"
            job["error"] = None
        except Exception as exc:
            job["state"] = "error"
            job["phase"] = "failed"
            job["error"] = f"{type(exc).__name__}: {exc}"[:1600]
        job["updated_at"] = _utc_now()
        job["finished_at"] = job["updated_at"]
        with self._lock:
            _atomic_write_json(path, job)

    def _recover_interrupted_jobs(self) -> None:
        for path in self.jobs_dir.glob("*.json"):
            try:
                job = _read_json(path)
            except RecordingExportError:
                continue
            if job.get("state") != "running":
                continue
            try:
                job["result"] = self._validate_output(job)
                job["state"] = "ready"
                job["phase"] = "published"
                job["error"] = None
                job["finished_at"] = _utc_now()
            except Exception:
                job["state"] = "queued"
                job["phase"] = "recovered"
                job["error"] = "worker interrupted before durable publication"
            job["updated_at"] = _utc_now()
            _atomic_write_json(path, job)

    @staticmethod
    def _public_job(job: Mapping[str, Any]) -> dict[str, Any]:
        source = job.get("source") if isinstance(job.get("source"), Mapping) else {}
        conversion = (
            job.get("conversion")
            if isinstance(job.get("conversion"), Mapping)
            else {}
        )
        output = job.get("output") if isinstance(job.get("output"), Mapping) else {}
        result = job.get("result") if isinstance(job.get("result"), Mapping) else {}
        episode_ids = source.get("episode_ids")
        return {
            "job_id": job.get("job_id"),
            "state": job.get("state"),
            "phase": job.get("phase"),
            "episode_count": int(source.get("episode_count") or 0),
            "episode_ids": list(episode_ids[:20]) if isinstance(episode_ids, list) else [],
            "episode_ids_truncated": bool(
                isinstance(episode_ids, list) and len(episode_ids) > 20
            ),
            "conversion_ready_count": int(conversion.get("ready_count") or 0),
            "conversion_states": dict(conversion.get("states") or {}),
            "dataset_name": output.get("dataset_name"),
            "dataset_root": output.get("dataset_root"),
            "video_materialization": dict(
                result.get("video_materialization") or {}
            ),
            "validation": dict(result.get("validation") or {}),
            "error": job.get("error"),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "finished_at": job.get("finished_at"),
        }

    def state(self, *, limit: int = 20) -> dict[str, Any]:
        jobs: list[dict[str, Any]] = []
        pending = False
        with self._lock:
            for path in self._ordered_job_paths(newest_first=True):
                try:
                    job = _read_json(path)
                except RecordingExportError:
                    continue
                pending = pending or job.get("state") in {"queued", "waiting"}
                jobs.append(self._public_job(job))
                if len(jobs) >= max(1, min(int(limit), 100)):
                    break
        if pending and self.auto_start and not self._closed:
            self.start()
            self._wake.set()
        counts = {state: 0 for state in sorted(VALID_STATES)}
        for job in jobs:
            if job.get("state") in counts:
                counts[str(job["state"])] += 1
        return {
            "schema_version": SCHEMA_VERSION,
            "configured": True,
            "available": self.export_python.is_file() and self.export_script.is_file(),
            "blockers": [
                message
                for available, message in (
                    (self.export_python.is_file(), f"Export Python not found: {self.export_python}"),
                    (self.export_script.is_file(), f"Export script not found: {self.export_script}"),
                )
                if not available
            ],
            "counts": counts,
            "jobs": jobs,
        }


__all__ = [
    "RecordingExportError",
    "RecordingExportManager",
]
