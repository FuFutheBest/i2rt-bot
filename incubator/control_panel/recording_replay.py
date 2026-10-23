"""Read-only discovery and inspection of configured NEDF rollout episodes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Optional
from datetime import datetime, timezone

from utils.runtime_profiles import select_runtime_python

from .converter_profile import (
    CONVERTER_FFMPEG_ENV,
    CONVERTER_FULL_DECODE_FFMPEG_ENV,
    CONVERTER_PYTHON_ENV,
    DEFAULT_PYTHON,
)
from .recording_replay_cache import (
    CacheEntry,
    DEFAULT_CACHE_MAX_BYTES,
    DEFAULT_CACHE_TTL_S,
    DEFAULT_SHADOW_ROOT,
    RecordingReplayCache,
    SourceIdentity,
)
from .recording_paths import (
    RECORDING_DATE_DIRECTORY,
    iter_recording_episode_roots,
)
from .recording_cameras import resolve_recorded_camera_ids
from .recording_contract import (
    NEDF3_DATA_FORMAT,
    camera_ids_for_format,
    normalize_data_format,
)


DEFAULT_INSPECTOR_PYTHON = DEFAULT_PYTHON
DEFAULT_INSPECTOR_SCRIPT = Path(__file__).resolve().parent / "tools/inspect_nedf3_episode.py"


def _default_inspector_python() -> Path:
    # Inspection uses the same SDK as the recorder.  A dedicated converter
    # environment is still supported for deployments that provide one, but a
    # launched Control Panel must be able to inspect its own recordings when
    # that optional release profile is absent.
    selection = select_runtime_python(
        "recording",
        environment_names=(
            "SUPERINFERENCE_RECORDING_PYTHON",
            "SUPERINFERENCE_RECORDER_PYTHON",
        ),
        config_name=os.environ.get("SUPERINFERENCE_CONFIG_NAME"),
        require_verified_profile=True,
    )
    if selection is not None:
        return selection.python
    configured_converter = os.environ.get(CONVERTER_PYTHON_ENV)
    if configured_converter and configured_converter.strip():
        return Path(configured_converter).expanduser()
    if DEFAULT_INSPECTOR_PYTHON.is_file() and os.access(DEFAULT_INSPECTOR_PYTHON, os.X_OK):
        return DEFAULT_INSPECTOR_PYTHON
    return Path(os.sys.executable).expanduser()


def _default_cache_root() -> Path:
    explicit = os.environ.get("SUPERINFERENCE_RECORDING_REPLAY_CACHE_ROOT")
    if isinstance(explicit, str) and explicit.strip():
        return Path(explicit.strip()).expanduser()
    data_root = os.environ.get("SUPERINFERENCE_DATA_ROOT")
    base = (
        Path(data_root.strip()).expanduser()
        if isinstance(data_root, str) and data_root.strip()
        else Path.home() / ".local/share/superinference-control-panel"
    )
    return base / "recording-replay"


def _default_replay_ffmpeg() -> Path:
    explicit = os.environ.get("SUPERINFERENCE_REPLAY_FFMPEG")
    if explicit:
        return Path(explicit).expanduser()
    full_decode = os.environ.get(CONVERTER_FULL_DECODE_FFMPEG_ENV)
    if full_decode:
        return Path(full_decode).expanduser()
    system_ffmpeg = Path("/usr/bin/ffmpeg")
    if system_ffmpeg.is_file():
        return system_ffmpeg
    converter_ffmpeg = os.environ.get(CONVERTER_FFMPEG_ENV)
    if converter_ffmpeg:
        return Path(converter_ffmpeg).expanduser()
    return system_ffmpeg


DEFAULT_CACHE_ROOT = _default_cache_root()
DEFAULT_REPLAY_FFMPEG = _default_replay_ffmpeg()
_AUTO_REPLAY_FFMPEG = object()
EPISODE_NAME = re.compile(r"^episode_[A-Za-z0-9_-]{1,120}$")
EPISODE_ID_SEPARATOR = ":"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_DENSE_BYTES = 4 * 1024 * 1024
MAX_DENSE_SEGMENTS = 5000
MAX_TRAJECTORY_POINTS = 5000
MAX_DELETE_EPISODES = 100
DEFAULT_BUILD_WAIT_S = 0.15
DEFAULT_BUILD_QUEUE_SIZE = 64
DEFAULT_BUILD_LOCK_WAIT_S = 2.0
DEFAULT_LIST_CACHE_TTL_S = 1.0
BUILD_STATE_STALE_S = 120.0
INSPECTOR_AS_LIMIT_BYTES = 1536 * 1024 * 1024
INSPECTOR_FILE_LIMIT_BYTES = 384 * 1024 * 1024
VIDEO_FEATURES = {
    "head": ("observation.images.image",),
    "left": (
        "observation.images.wrist_image_1",
        "observation.images.left_wrist",
    ),
    "right": (
        "observation.images.wrist_image_2",
        "observation.images.right_wrist",
    ),
}


class RecordingReplayError(RuntimeError):
    """Base error for recording replay requests."""


class RecordingReplayRequestError(RecordingReplayError):
    """A caller supplied an invalid public parameter."""


class RecordingReplayNotFound(RecordingReplayError):
    """The requested episode does not exist under the configured root."""


class RecordingReplayUnavailable(RecordingReplayError):
    """The episode cannot provide a truthful joint trajectory."""

    def __init__(
        self,
        message: str,
        *,
        availability: Optional[Mapping[str, Any]] = None,
        code: str = "joint_trajectory_unavailable",
    ) -> None:
        super().__init__(message)
        self.availability = dict(availability or {})
        self.code = str(code)


class RecordingReplayBuilding(RecordingReplayError):
    """A bounded background build is in progress for this episode."""

    def __init__(self, episode_id: str, retry_after_ms: int = 250) -> None:
        super().__init__("Recording replay cache is building")
        self.episode_id = episode_id
        self.retry_after_ms = max(50, int(retry_after_ms))


class RecordingReplayBuildFailed(RecordingReplayError):
    """A cache build failed and is held by persisted backoff."""

    def __init__(self, episode_id: str, error: str, retry_after_ms: int) -> None:
        super().__init__(error)
        self.episode_id = episode_id
        self.retry_after_ms = max(50, int(retry_after_ms))


@dataclass(frozen=True)
class _BuildJob:
    episode_id: str
    path: Path
    identity: SourceIdentity
    entry: CacheEntry
    camera_roles: dict[str, Optional[str]]
    inspector_python: Path


class _BuildDeferred(RuntimeError):
    """Another process owns this generation; do not overwrite shared state."""


@dataclass(frozen=True)
class _FinalizedHint:
    path: Path
    result: str


def _episode_id_parts(episode_id: str) -> Optional[tuple[Optional[str], str]]:
    if not isinstance(episode_id, str) or not episode_id:
        return None
    parts = episode_id.split(EPISODE_ID_SEPARATOR)
    if len(parts) == 1 and EPISODE_NAME.fullmatch(parts[0]):
        return None, parts[0]
    if (
        len(parts) == 2
        and RECORDING_DATE_DIRECTORY.fullmatch(parts[0])
        and EPISODE_NAME.fullmatch(parts[1])
    ):
        return parts[0], parts[1]
    return None


def _public_episode_id(date_name: Optional[str], episode_name: str) -> str:
    return (
        f"{date_name}{EPISODE_ID_SEPARATOR}{episode_name}"
        if date_name is not None
        else episode_name
    )


def _episode_display_name(date_name: Optional[str], episode_name: str) -> str:
    return f"{date_name} / {episode_name}" if date_name else episode_name


def _safe_json(path: Path, *, max_bytes: int = 1024 * 1024) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError:
        return {}
    if path.is_symlink() or not path.is_file() or info.st_size <= 0 or info.st_size > max_bytes:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _safe_text(value: Any, limit: int = 300) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value[:limit]


def _safe_int(value: Any, *, minimum: int = 0) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return int(value)


def _status_for(
    path: Path,
    result: Mapping[str, Any],
    incomplete: Mapping[str, Any],
    marker: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> str:
    if path.name.endswith("_temp"):
        return "recording"
    decision_status = str(decision.get("status") or "").lower()
    if decision_status == "accepted":
        return "accepted"
    if decision_status == "rejected":
        return "rejected"
    if decision or marker.get("decision_required") is True:
        return "incomplete"
    if incomplete or "_incomplete_" in path.name:
        return "incomplete"
    episode_result = str(result.get("episode_result") or "").lower()
    if episode_result == "success":
        return "accepted"
    if episode_result == "failure":
        return "rejected"
    return "unfinalized"


def _dense_segments(path: Path) -> tuple[bool, list[dict[str, Any]], list[dict[str, str]]]:
    dense_path = path / "dense_labels.jsonl"
    try:
        info = dense_path.lstat()
    except OSError:
        return False, [], []
    if dense_path.is_symlink() or not dense_path.is_file() or info.st_size > MAX_DENSE_BYTES:
        return True, [], [{"code": "invalid_dense_labels", "message": "Dense label sidecar is unsafe or too large", "severity": "warning"}]
    segments: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    try:
        with dense_path.open("r", encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if index >= MAX_DENSE_SEGMENTS:
                    issues.append({"code": "dense_labels_truncated", "message": "Dense label list exceeds the inspection limit", "severity": "warning"})
                    break
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    issues.append({"code": "malformed_dense_label", "message": "A dense label entry is malformed", "severity": "warning"})
                    continue
                if not isinstance(raw, Mapping) or raw.get("label") not in {"accept", "reject"}:
                    issues.append({"code": "invalid_dense_label", "message": "A dense label entry has an invalid schema", "severity": "warning"})
                    continue
                start = _safe_int(raw.get("start_frame"))
                end = _safe_int(raw.get("end_frame"))
                if start is None or end is None or end < start:
                    issues.append({"code": "invalid_dense_range", "message": "A dense label entry has an invalid frame range", "severity": "warning"})
                    continue
                segments.append({
                    "label": raw["label"],
                    "start_frame": start,
                    "end_frame": end,
                    "start_timestamp_ns": _safe_int(raw.get("start_timestamp_ns")),
                    "end_timestamp_ns": _safe_int(raw.get("end_timestamp_ns")),
                    "checkpoint_revision": _safe_int(raw.get("checkpoint_revision")),
                })
    except (OSError, UnicodeError):
        issues.append({"code": "unreadable_dense_labels", "message": "Dense label sidecar cannot be read", "severity": "warning"})
    return True, segments, issues


class RecordingReplayService:
    """Bounded facade around one trusted SDK subprocess and stat-keyed cache."""

    def __init__(
        self,
        root_provider: Callable[[], Optional[Path]],
        *,
        camera_roles_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
        published_shard_provider: Optional[Callable[[Path], Optional[Path]]] = None,
        inspector_python: Optional[Path] = None,
        inspector_python_provider: Optional[Callable[[], Optional[Path]]] = None,
        inspector_script: Path = DEFAULT_INSPECTOR_SCRIPT,
        cache_root: Path = DEFAULT_CACHE_ROOT,
        replay_ffmpeg: Any = _AUTO_REPLAY_FFMPEG,
        process_runner: Callable[..., Any] = subprocess.run,
        timeout_s: float = 20.0,
        cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
        cache_max_bytes: int = DEFAULT_CACHE_MAX_BYTES,
        build_wait_s: float = DEFAULT_BUILD_WAIT_S,
        build_workers: int = 1,
        build_queue_size: int = DEFAULT_BUILD_QUEUE_SIZE,
        list_cache_ttl_s: float = DEFAULT_LIST_CACHE_TTL_S,
        shadow_root: Path = DEFAULT_SHADOW_ROOT,
    ) -> None:
        self._root_provider = root_provider
        self._camera_roles_provider = camera_roles_provider
        self._published_shard_provider = published_shard_provider
        self._inspector_python_provider = inspector_python_provider
        self.inspector_python = Path(
            inspector_python
            if inspector_python is not None
            else _default_inspector_python()
        ).expanduser().absolute()
        self.inspector_script = Path(inspector_script)
        self.cache_root = Path(cache_root)
        selected_replay_ffmpeg = (
            _default_replay_ffmpeg()
            if replay_ffmpeg is _AUTO_REPLAY_FFMPEG
            else replay_ffmpeg
        )
        self.replay_ffmpeg = (
            Path(selected_replay_ffmpeg).expanduser().resolve(strict=False)
            if selected_replay_ffmpeg is not None
            else None
        )
        self._process_runner = process_runner
        self.timeout_s = max(1.0, min(float(timeout_s), 60.0))
        self.build_wait_s = max(0.0, min(float(build_wait_s), 0.5))
        self.list_cache_ttl_s = max(0.0, min(float(list_cache_ttl_s), 5.0))
        self._cache = RecordingReplayCache(
            self.cache_root,
            ttl_s=cache_ttl_s,
            max_bytes=cache_max_bytes,
            shadow_root=shadow_root,
        )
        self._build_queue: queue.Queue[_BuildJob | _FinalizedHint] = queue.Queue(
            maxsize=max(1, min(int(build_queue_size), 1024))
        )
        self._build_guard = threading.Lock()
        self._finalized_guard = threading.Lock()
        self._list_guard = threading.Lock()
        self._delete_guard = threading.Lock()
        self._list_cache_root: Optional[str] = None
        self._list_cache_expires_at = 0.0
        self._list_cache_items: tuple[dict[str, Any], ...] = ()
        self._build_events: dict[str, threading.Event] = {}
        self._finalized_pending: set[str] = set()
        self._build_threads: list[threading.Thread] = []
        for index in range(max(1, min(int(build_workers), 4))):
            worker = threading.Thread(
                target=self._build_worker,
                name=f"recording-replay-cache-{index}",
                daemon=True,
            )
            worker.start()
            self._build_threads.append(worker)

    def _effective_inspector_python(self) -> Path:
        if self._inspector_python_provider is not None:
            configured = self._inspector_python_provider()
            if configured is not None:
                return Path(configured).expanduser().absolute()
        return self.inspector_python

    def _profile_fingerprint(
        self,
        camera_roles: Mapping[str, Any],
        *,
        inspector_python: Optional[Path] = None,
    ) -> str:
        runtime_profile = dict(camera_roles)
        runtime_profile["__inspector_python"] = str(
            inspector_python or self._effective_inspector_python()
        )
        runtime_profile["__replay_ffmpeg"] = (
            str(self.replay_ffmpeg) if self.replay_ffmpeg is not None else None
        )
        runtime_profile["__replay_runtime_schema"] = 3
        return self._cache.profile_fingerprint(runtime_profile)

    def _root(self) -> Optional[Path]:
        root = self._root_provider()
        if root is None:
            return None
        return Path(root).expanduser().resolve(strict=False)

    def _episode(self, episode_id: str) -> Path:
        parsed = _episode_id_parts(episode_id)
        if parsed is None:
            raise RecordingReplayRequestError("Invalid episode id")
        date_name, episode_name = parsed
        root = self._root()
        if root is None or not root.is_dir():
            raise RecordingReplayNotFound("Recording root is not available")
        candidate = root / episode_name if date_name is None else root / date_name / episode_name
        try:
            info = candidate.lstat()
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise RecordingReplayNotFound("Recording episode was not found") from exc
        if candidate.is_symlink() or not candidate.is_dir() or not info:
            raise RecordingReplayNotFound("Recording episode was not found")
        return resolved

    @staticmethod
    def _episode_metadata(path: Path) -> dict[str, Any]:
        metadata = _safe_json(path / "metadata.json")
        result = _safe_json(path / "episode_result.json")
        incomplete = _safe_json(path / "episode_incomplete.json")
        marker = _safe_json(path / ".conversion_source.json")
        decision = _safe_json(path / ".recording_decision.json")
        try:
            updated_ns = path.stat().st_mtime_ns
        except OSError:
            updated_ns = 0
        try:
            data_format = normalize_data_format(metadata.get("data_format"))
        except ValueError:
            data_format = None
        return {
            "id": path.name,
            "name": path.name,
            "status": _status_for(path, result, incomplete, marker, decision),
            "result": _safe_text(result.get("episode_result"), 40) or _safe_text(incomplete.get("episode_result"), 40),
            "episode_index": _safe_int(result.get("episode_index")) if result else _safe_int(incomplete.get("episode_index")),
            "episode_id": _safe_text(result.get("episode_id"), 160) if result else _safe_text(incomplete.get("episode_id"), 160),
            "task_name": _safe_text(metadata.get("task_name"), 300),
            "data_format": data_format,
            "config_name": _safe_text(marker.get("config_name"), 160),
            "updated_at": (
                datetime.fromtimestamp(updated_ns / 1_000_000_000, tz=timezone.utc).isoformat()
                if updated_ns > 0
                else None
            ),
            "updated_at_ns": updated_ns,
            "has_metadata": bool(metadata),
            "has_dense_labels": (path / "dense_labels.jsonl").is_file() and not (path / "dense_labels.jsonl").is_symlink(),
            "has_conversion_check": bool(marker),
            "conversion_check_status": "published" if marker else "not_published",
        }

    def list(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        status: str = "all",
        result: str = "all",
        date: str = "all",
        query: str = "",
    ) -> dict[str, Any]:
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise RecordingReplayRequestError("page must be a positive integer")
        if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 100:
            raise RecordingReplayRequestError("page_size must be between 1 and 100")
        allowed_status = {"all", "accepted", "rejected", "incomplete", "recording", "unfinalized"}
        if status not in allowed_status:
            raise RecordingReplayRequestError("Invalid recording status filter")
        allowed_results = {"all", "success", "failure", "incomplete", "unknown"}
        if result not in allowed_results:
            raise RecordingReplayRequestError("Invalid recording result filter")
        selected_date = str(date or "all").strip()
        if selected_date not in {"all", "legacy"} and RECORDING_DATE_DIRECTORY.fullmatch(
            selected_date
        ) is None:
            raise RecordingReplayRequestError("Invalid recording date filter")
        search = str(query or "").strip()
        if len(search) > 120:
            raise RecordingReplayRequestError("Recording search query is too long")
        search_folded = search.casefold()
        root = self._root()
        items: list[dict[str, Any]] = []
        date_rows: list[dict[str, Any]] = []
        root_state = "unconfigured" if root is None else "missing"
        if root is not None and root.is_dir():
            root_state = "ready"
            try:
                indexed = self._recording_index(root)
            except (OSError, ValueError):
                indexed = []
                root_state = "unreadable"
            if not indexed and root_state == "ready":
                root_state = "empty"
            dates: dict[str, dict[str, Any]] = {}
            for item in indexed:
                date_id = str(item.get("date") or "legacy")
                row = dates.setdefault(
                    date_id,
                    {
                        "id": date_id,
                        "label": (
                            "历史（未分日期）" if date_id == "legacy" else date_id
                        ),
                        "count": 0,
                        "accepted": 0,
                        "rejected": 0,
                        "latest_updated_at_ns": 0,
                    },
                )
                row["count"] += 1
                item_status = str(item.get("status") or "")
                if item_status in {"accepted", "rejected"}:
                    row[item_status] += 1
                row["latest_updated_at_ns"] = max(
                    int(row["latest_updated_at_ns"]),
                    int(item.get("updated_at_ns") or 0),
                )
            date_rows = sorted(
                dates.values(),
                key=lambda row: (
                    row["id"] != "legacy",
                    row["id"] if row["id"] != "legacy" else "",
                ),
                reverse=True,
            )
            items = []
            for item in indexed:
                item_date = str(item.get("date") or "legacy")
                if selected_date != "all" and item_date != selected_date:
                    continue
                if status != "all" and item.get("status") != status:
                    continue
                if result != "all" and item.get("result") != result:
                    continue
                if search_folded:
                    searchable = " ".join(
                        str(value or "")
                        for value in (
                            item.get("id"),
                            item.get("name"),
                            item.get("episode_name"),
                            item.get("episode_id"),
                            item.get("episode_index"),
                            item.get("task_name"),
                        )
                    ).casefold()
                    if search_folded not in searchable:
                        continue
                items.append(item)
        items.sort(key=lambda item: (item["updated_at_ns"], item["id"]), reverse=True)
        total = len(items)
        offset = (page - 1) * page_size
        return {
            "items": items[offset : offset + page_size],
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": math.ceil(total / page_size) if total else 0,
            "root_configured": root is not None,
            "root_available": bool(root and root.is_dir()),
            "root_state": root_state,
            "dates": date_rows,
            "filters": {
                "date": selected_date,
                "status": status,
                "result": result,
                "query": search,
            },
            "cache": {
                "persistent": True,
                "root_name": self.cache_root.name,
            },
        }

    def delete(self, episode_ids: Any) -> dict[str, Any]:
        if isinstance(episode_ids, (str, bytes)) or not isinstance(
            episode_ids, (list, tuple)
        ):
            raise RecordingReplayRequestError("episode_ids must be an array")
        requested = list(episode_ids)
        if not requested:
            raise RecordingReplayRequestError("Select at least one episode to delete")
        if len(requested) > MAX_DELETE_EPISODES:
            raise RecordingReplayRequestError(
                f"At most {MAX_DELETE_EPISODES} episodes may be deleted at once"
            )
        if any(not isinstance(value, str) for value in requested):
            raise RecordingReplayRequestError("episode_ids must contain strings")
        if len(set(requested)) != len(requested):
            raise RecordingReplayRequestError("episode_ids contains duplicates")

        with self._delete_guard:
            root = self._root()
            if root is None or not root.is_dir():
                raise RecordingReplayNotFound("Recording root is not available")
            root = root.resolve(strict=True)
            targets: list[tuple[str, Path, int]] = []
            for episode_id in requested:
                path = self._episode(episode_id)
                metadata = self._episode_metadata(path)
                if metadata.get("status") == "recording":
                    raise RecordingReplayRequestError(
                        f"Active recording cannot be deleted: {episode_id}"
                    )
                size_bytes = 0
                for current, directories, files in os.walk(path, followlinks=False):
                    directories[:] = [
                        name for name in directories
                        if not (Path(current) / name).is_symlink()
                    ]
                    for name in files:
                        try:
                            size_bytes += (Path(current) / name).lstat().st_size
                        except OSError:
                            pass
                targets.append((episode_id, path, size_bytes))

            trash = root / (
                f".control-panel-delete-{os.getpid()}-{threading.get_ident()}-{time.time_ns()}"
            )
            moved: list[tuple[Path, Path]] = []
            try:
                trash.mkdir(mode=0o700)
                for index, (_episode_id, source, _size) in enumerate(targets):
                    destination = trash / f"{index:03d}-{source.name}"
                    os.replace(source, destination)
                    moved.append((source, destination))
            except OSError as exc:
                for source, destination in reversed(moved):
                    try:
                        if destination.exists() and not source.exists():
                            os.replace(destination, source)
                    except OSError:
                        pass
                try:
                    trash.rmdir()
                except OSError:
                    pass
                raise RecordingReplayUnavailable(
                    "Recording deletion could not be staged"
                ) from exc

            for _episode_id, source, _size in targets:
                self._cache.remove_episode(source)
            try:
                shutil.rmtree(trash)
            except OSError as exc:
                raise RecordingReplayUnavailable(
                    "Recording deletion was staged but cleanup is incomplete"
                ) from exc
            for _episode_id, source, _size in targets:
                parent = source.parent
                if parent != root and RECORDING_DATE_DIRECTORY.fullmatch(parent.name):
                    try:
                        parent.rmdir()
                    except OSError:
                        pass
            with self._list_guard:
                self._list_cache_root = None
                self._list_cache_expires_at = 0.0
                self._list_cache_items = ()
            return {
                "ok": True,
                "deleted": [episode_id for episode_id, _path, _size in targets],
                "deleted_count": len(targets),
                "freed_bytes": sum(size for _episode_id, _path, size in targets),
            }

    def _recording_index(self, root: Path) -> list[dict[str, Any]]:
        cache_root = str(root.resolve(strict=False))
        with self._list_guard:
            now = time.monotonic()
            if (
                self._list_cache_root == cache_root
                and now < self._list_cache_expires_at
            ):
                return [dict(item) for item in self._list_cache_items]

            indexed: list[dict[str, Any]] = []
            for episode_root in iter_recording_episode_roots(root):
                date_name = None if episode_root == root else episode_root.name
                root_items = self._cache.episode_index(
                    episode_root,
                    name_is_valid=lambda name: EPISODE_NAME.fullmatch(name) is not None,
                    metadata_loader=self._episode_metadata,
                )
                for item in root_items:
                    episode_name = str(item.get("id") or "")
                    if EPISODE_NAME.fullmatch(episode_name) is None:
                        continue
                    item["id"] = _public_episode_id(date_name, episode_name)
                    item["name"] = _episode_display_name(date_name, episode_name)
                    item["episode_name"] = episode_name
                    item["date"] = date_name
                    indexed.append(item)
            indexed.sort(
                key=lambda item: (item["updated_at_ns"], item["id"]),
                reverse=True,
            )
            self._list_cache_root = cache_root
            self._list_cache_items = tuple(dict(item) for item in indexed)
            self._list_cache_expires_at = now + self.list_cache_ttl_s
            return [dict(item) for item in indexed]

    @staticmethod
    def _job_key(entry: CacheEntry) -> str:
        return str(entry.directory)

    @staticmethod
    def _lower_worker_priority() -> None:
        try:
            os.nice(19)
        except OSError:
            pass
        try:
            if hasattr(os, "SCHED_IDLE"):
                os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
        except (AttributeError, OSError, PermissionError):
            pass
        try:
            subprocess.run(
                ["/usr/bin/ionice", "-c", "3", "-p", str(threading.get_native_id())],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    def _build_worker(self) -> None:
        self._lower_worker_priority()
        while True:
            queued = self._build_queue.get()
            if isinstance(queued, _FinalizedHint):
                hint_key = str(queued.path)
                try:
                    try:
                        self._process_finalized_hint(queued)
                    except Exception:
                        # Finalize prefetch is advisory. Lazy HTTP discovery
                        # remains the recovery path after any worker failure.
                        pass
                finally:
                    with self._finalized_guard:
                        self._finalized_pending.discard(hint_key)
                    self._build_queue.task_done()
                continue
            job = queued
            key = self._job_key(job.entry)
            previous = self._cache.read_build_state(job.entry)
            attempts = int(previous.get("attempts") or 0)
            try:
                if previous.get("status") != "ready":
                    self._cache.write_build_state(job.entry, "building", attempts=attempts)
                self._build_one(job)
                self._cache.write_build_state(job.entry, "ready", attempts=0)
            except _BuildDeferred:
                # The lock owner retains authority over persisted building/ready
                # state.  A losing process must not publish a false failure.
                pass
            except BaseException as exc:
                attempts += 1
                delay_s = min(300, max(1, 2 ** min(attempts - 1, 8)))
                if isinstance(exc, RecordingReplayUnavailable):
                    error = str(exc)
                    code = "inspection_unavailable"
                elif isinstance(exc, ValueError):
                    error = "Recording replay cache build failed"
                    code = str(exc)[:100]
                else:
                    error = "Recording replay cache build failed"
                    code = "unexpected_build_failure"
                self._cache.write_build_state(
                    job.entry,
                    "failed",
                    attempts=attempts,
                    retry_at_ns=time.time_ns() + int(delay_s * 1e9),
                    code=code,
                    error=error,
                )
            finally:
                with self._build_guard:
                    event = self._build_events.pop(key, None)
                if event is not None:
                    event.set()
                self._build_queue.task_done()

    def enqueue_finalized(self, receipt: Mapping[str, Any]) -> bool:
        """Queue one stable finalized episode without touching the filesystem."""
        if not isinstance(receipt, Mapping):
            return False
        raw_path = receipt.get("path")
        result = receipt.get("result")
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or len(raw_path) > 4096
            or "\x00" in raw_path
            or result != "success"
        ):
            return False
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            return False
        key = str(path)
        with self._finalized_guard:
            if key in self._finalized_pending:
                return True
            self._finalized_pending.add(key)
            try:
                self._build_queue.put_nowait(_FinalizedHint(path, str(result)))
            except queue.Full:
                self._finalized_pending.discard(key)
                return False
        return True

    def _process_finalized_hint(self, hint: _FinalizedHint) -> None:
        try:
            root = self._root()
            if root is None:
                return
            root = root.resolve(strict=True)
            path = hint.path.resolve(strict=True)
            relative = path.relative_to(root)
            if not EPISODE_NAME.fullmatch(path.name) or path.is_symlink() or not path.is_dir():
                return
            relative_parts = relative.parts
            if len(relative_parts) == 1:
                episode_id = path.name
            elif (
                len(relative_parts) == 2
                and RECORDING_DATE_DIRECTORY.fullmatch(relative_parts[0])
            ):
                episode_id = _public_episode_id(relative_parts[0], path.name)
            else:
                return
            camera_roles = self._expected_camera_roles()
            identity = self._cache.source_identity(path)
            inspector_python = self._effective_inspector_python()
        except (OSError, ValueError):
            return
        job = _BuildJob(
            episode_id,
            path,
            identity,
            self._cache.entry(
                path,
                identity.fingerprint,
                self._profile_fingerprint(
                    camera_roles,
                    inspector_python=inspector_python,
                ),
            ),
            camera_roles,
            inspector_python,
        )
        self._enqueue_build(job)

    def _enqueue_build(self, job: _BuildJob) -> Optional[threading.Event]:
        now_ns = time.time_ns()
        state = self._cache.read_build_state(job.entry)
        complete_unverified_generation = (
            self._cache.load(job.entry, verify_hash=False).status == "unverified"
        )
        if state.get("status") == "failed" and int(state.get("retry_at_ns") or 0) > now_ns:
            return None
        if state.get("status") == "building" and not complete_unverified_generation:
            updated_ns = int(state.get("updated_at_ns") or 0)
            if updated_ns and now_ns - updated_ns < int(BUILD_STATE_STALE_S * 1e9):
                with self._build_guard:
                    return self._build_events.get(self._job_key(job.entry))
        key = self._job_key(job.entry)
        with self._build_guard:
            existing = self._build_events.get(key)
            if existing is not None:
                return existing
            event = threading.Event()
            self._build_events[key] = event
            try:
                if state.get("status") != "ready":
                    self._cache.write_build_state(
                        job.entry,
                        "building",
                        attempts=int(state.get("attempts") or 0),
                    )
                self._build_queue.put_nowait(job)
            except queue.Full:
                self._build_events.pop(key, None)
                self._cache.write_build_state(
                    job.entry,
                    "failed",
                    attempts=int(state.get("attempts") or 0) + 1,
                    retry_at_ns=time.time_ns() + 1_000_000_000,
                    code="build_queue_full",
                    error="Recording replay cache queue is full",
                )
                event.set()
            return event

    def _build_one(self, job: _BuildJob) -> None:
        with self._cache.entry_lock(job.entry, timeout_s=DEFAULT_BUILD_LOCK_WAIT_S) as acquired:
            if not acquired:
                if self._cache.load(job.entry, verify_hash=False).status == "ready":
                    return
                raise _BuildDeferred("recording_generation_owned_elsewhere")
            load = self._cache.load(job.entry, verify_hash=True)
            if load.status == "ready":
                return
            if load.status == "corrupt":
                self._cache.quarantine_locked(job.entry, reason="integrity_validation_failed")
            elif load.status == "expired":
                self._cache.remove_complete_locked(job.entry)
            try:
                current_identity = self._cache.source_identity(job.path)
            except (OSError, ValueError) as exc:
                raise RecordingReplayUnavailable("Recording source identity is unavailable") from exc
            if current_identity.fingerprint != job.identity.fingerprint:
                raise RecordingReplayUnavailable("Recording changed before inspection")

            staging, timeline_output = self._cache.prepare_staging(job.entry)
            try:
                with self._cache.shadow_episode(job.path, job.identity) as shadow:
                    cpu_soft = max(2, int(math.ceil(self.timeout_s)) + 1)
                    cpu_hard = cpu_soft + 1
                    command = [
                        "/usr/bin/prlimit",
                        f"--cpu={cpu_soft}:{cpu_hard}",
                        f"--as={INSPECTOR_AS_LIMIT_BYTES}:{INSPECTOR_AS_LIMIT_BYTES}",
                        f"--fsize={INSPECTOR_FILE_LIMIT_BYTES}:{INSPECTOR_FILE_LIMIT_BYTES}",
                        "--nofile=128:128",
                        "--",
                        "/usr/bin/nice", "-n", "19",
                        "/usr/bin/ionice", "-c", "3",
                        str(job.inspector_python), str(self.inspector_script),
                        "--episode", str(shadow),
                        "--mode", "snapshot",
                        "--timeline-output", str(timeline_output),
                    ]
                    if self.replay_ffmpeg is not None:
                        command.extend(
                            [
                                "--video-output-dir",
                                str(staging),
                                "--ffmpeg-bin",
                                str(self.replay_ffmpeg),
                            ]
                        )
                        for role in ("head", "left", "right"):
                            camera_id = job.camera_roles.get(role)
                            if camera_id:
                                command.extend(
                                    ["--camera-role", f"{role}={camera_id}"]
                                )
                    try:
                        completed = self._process_runner(
                            command,
                            capture_output=True,
                            text=True,
                            timeout=self.timeout_s,
                            check=False,
                            env={
                                "PATH": "/usr/bin:/bin",
                                "PYTHONNOUSERSITE": "1",
                                "OPENBLAS_NUM_THREADS": "1",
                                "OMP_NUM_THREADS": "1",
                                "MKL_NUM_THREADS": "1",
                                "NUMEXPR_NUM_THREADS": "1",
                            },
                        )
                    except subprocess.TimeoutExpired as exc:
                        raise RecordingReplayUnavailable("Recording inspection timed out") from exc
                    except OSError as exc:
                        raise RecordingReplayUnavailable("Recording inspection runtime is unavailable") from exc
                if completed.returncode != 0:
                    # Keep SDK stderr out of the API response: it may contain
                    # local paths or device details.  The exit code is useful
                    # for correlating the bounded worker log without leaking
                    # that subprocess output to the browser.
                    raise RecordingReplayUnavailable(
                        "NEDF recording could not be inspected "
                        f"(inspector exit {int(completed.returncode)})"
                    )
                if len(completed.stdout.encode("utf-8", errors="ignore")) > MAX_JSON_BYTES:
                    raise RecordingReplayUnavailable("Recording inspection result is too large")
                try:
                    inspection = json.loads(completed.stdout)
                except json.JSONDecodeError as exc:
                    raise RecordingReplayUnavailable("Recording inspector returned malformed data") from exc
                if not isinstance(inspection, dict) or inspection.get("schema_version") != 1:
                    raise RecordingReplayUnavailable("Recording inspector returned an unsupported schema")
                try:
                    final_identity = self._cache.source_identity(job.path)
                except (OSError, ValueError) as exc:
                    raise RecordingReplayUnavailable("Recording changed during inspection") from exc
                if final_identity.fingerprint != job.identity.fingerprint:
                    raise RecordingReplayUnavailable("Recording changed during inspection")
                try:
                    self._cache.publish(job.entry, inspection=inspection, staging=staging)
                except (OSError, ValueError) as exc:
                    raise RecordingReplayUnavailable("Recording cache could not be published") from exc
            finally:
                self._cache.discard_staging(staging)
        self._cache.cleanup(keep=job.entry)

    def _snapshot(
        self,
        path: Path,
        episode_id: Optional[str] = None,
    ) -> tuple[dict[str, Any], CacheEntry, dict[str, Optional[str]]]:
        public_id = episode_id or path.name
        camera_roles = self._expected_camera_roles()
        inspector_python = self._effective_inspector_python()
        try:
            identity = self._cache.source_identity(path)
        except (OSError, ValueError) as exc:
            raise RecordingReplayBuildFailed(
                public_id,
                "Recording source identity is unavailable",
                1000,
            ) from exc
        entry = self._cache.entry(
            path,
            identity.fingerprint,
            self._profile_fingerprint(
                camera_roles,
                inspector_python=inspector_python,
            ),
        )
        load = self._cache.load(entry, verify_hash=False)
        if load.status == "ready" and load.inspection is not None:
            return load.inspection, entry, camera_roles
        job = _BuildJob(
            public_id,
            path,
            identity,
            entry,
            camera_roles,
            inspector_python,
        )
        event = self._enqueue_build(job)
        deadline = time.monotonic() + self.build_wait_s
        while self.build_wait_s > 0 and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if event is not None:
                event.wait(timeout=max(0.0, remaining))
            else:
                time.sleep(min(0.02, max(0.0, remaining)))
            load = self._cache.load(entry, verify_hash=False)
            if load.status == "ready" and load.inspection is not None:
                return load.inspection, entry, camera_roles
            if event is not None and event.is_set():
                break
        load = self._cache.load(entry, verify_hash=False)
        if load.status == "ready" and load.inspection is not None:
            return load.inspection, entry, camera_roles
        state = self._cache.read_build_state(entry)
        retry_at_ns = int(state.get("retry_at_ns") or 0)
        if state.get("status") == "failed" and retry_at_ns > time.time_ns():
            retry_after_ms = max(50, math.ceil((retry_at_ns - time.time_ns()) / 1_000_000))
            raise RecordingReplayBuildFailed(
                path.name,
                str(state.get("error") or "Recording replay cache build failed"),
                retry_after_ms,
            )
        raise RecordingReplayBuilding(path.name)

    def _inspect(self, path: Path) -> dict[str, Any]:
        return self._snapshot(path)[0]

    def _expected_camera_roles(self) -> dict[str, Optional[str]]:
        raw: Mapping[str, Any] = {}
        if self._camera_roles_provider is not None:
            try:
                candidate = self._camera_roles_provider()
            except (TypeError, ValueError):
                candidate = {}
            if isinstance(candidate, Mapping):
                raw = candidate
        result: dict[str, Optional[str]] = {}
        for role in ("head", "left", "right"):
            value = raw.get(role, raw.get(f"{role}_camera"))
            if isinstance(value, Mapping):
                value = value.get("device_key", value.get("device"))
            result[role] = value.strip() if isinstance(value, str) and value.strip() else None
        return result

    @staticmethod
    def _topic_for_camera(
        inspection: Mapping[str, Any],
        device_key: Optional[str],
    ) -> Optional[str]:
        if device_key is None:
            return None
        token = re.sub(r"[^a-z0-9]+", "_", device_key.casefold()).strip("_")
        candidates: list[tuple[int, str]] = []
        rows = inspection.get("topics")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                name = _safe_text(row.get("name"))
                if not name:
                    continue
                lowered = name.casefold()
                is_nedf3_camera = (
                    lowered.startswith("nmx/hal/camera/")
                    and lowered.endswith("/rgbd")
                )
                if not is_nedf3_camera:
                    continue
                normalized = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
                if (
                    normalized.endswith(token)
                    or normalized.endswith(f"{token}_rgbd")
                ):
                    candidates.append((_safe_int(row.get("count")) or 0, name))
        return max(candidates, default=(0, ""))[1] or None

    def _camera_roles(
        self,
        inspection: Mapping[str, Any],
        expected: Optional[Mapping[str, Optional[str]]] = None,
    ) -> dict[str, dict[str, Any]]:
        cameras: dict[str, Mapping[str, Any]] = {}
        recorded_camera_info: dict[str, str] = {}
        rows = inspection.get("cameras")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                camera_id = _safe_text(row.get("id"))
                if camera_id:
                    cameras[camera_id] = row
                    source_device = _safe_text(row.get("source_device"))
                    if source_device:
                        recorded_camera_info[camera_id] = source_device

        result: dict[str, dict[str, Any]] = {}
        expected_roles = dict(expected or self._expected_camera_roles())
        try:
            data_format = normalize_data_format(inspection.get("data_format"))
        except ValueError:
            data_format = NEDF3_DATA_FORMAT
        resolved_ids = resolve_recorded_camera_ids(
            {
                camera_id: recorded_camera_info.get(camera_id, camera_id)
                for camera_id in cameras
            },
            expected_roles,
            data_format=data_format,
        )
        logical_camera_ids = camera_ids_for_format(data_format)
        for role, expected_device_key in expected_roles.items():
            camera_id = resolved_ids.get(role)
            row = cameras.get(camera_id) if camera_id is not None else None
            count = _safe_int(row.get("count")) if row is not None else 0
            topic = self._topic_for_camera(inspection, camera_id)
            logical_id = logical_camera_ids[role]
            source_device = (
                _safe_text(row.get("source_device")) if row is not None else None
            )
            device_key = source_device or expected_device_key
            if camera_id is None:
                status = "unknown"
                reason = "camera mapping missing; LeRobot conversion unavailable"
            elif row is None:
                status = "missing"
                reason = "recorded camera topic is absent"
            elif row.get("black_placeholder") is True:
                status = "black_placeholder"
                reason = "recorded camera contains black placeholder frames"
            elif count and count > 0 and topic:
                status = "healthy"
                reason = None
            elif count and count > 0:
                status = "missing"
                reason = "recorded camera topic could not be resolved"
            else:
                status = "missing"
                reason = "recorded camera topic contains no frames"
            result[role] = {
                "role": role,
                "camera_id": camera_id,
                "device_key": device_key,
                "topic": topic,
                "shm_name": f"{device_key}_data" if device_key else None,
                "status": status,
                "reason": reason,
                "count": count or 0,
                "black_placeholder": status == "black_placeholder",
            }
        return result

    @staticmethod
    def _shard_video(
        shard: Path, role: str
    ) -> tuple[Optional[Path], Optional[str]]:
        features = VIDEO_FEATURES.get(role)
        if features is None:
            raise RecordingReplayRequestError("Invalid recording video role")
        try:
            root = Path(shard).resolve(strict=True)
        except OSError:
            return None, None
        for feature in features:
            try:
                candidate = (
                    root
                    / "videos"
                    / "chunk-000"
                    / feature
                    / "episode_000000.mp4"
                )
                info = candidate.lstat()
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError):
                continue
            if (
                not candidate.is_symlink()
                and candidate.is_file()
                and info.st_size > 0
            ):
                return resolved, feature
        return None, None

    @staticmethod
    def _cached_video(
        entry: CacheEntry,
        inspection: Mapping[str, Any],
        role: str,
    ) -> tuple[Optional[Path], Optional[str]]:
        if role not in VIDEO_FEATURES:
            raise RecordingReplayRequestError("Invalid recording video role")
        videos = inspection.get("replay_videos")
        raw = videos.get(role) if isinstance(videos, Mapping) else None
        if not isinstance(raw, Mapping) or raw.get("available") is not True:
            return None, None
        expected_name = f"video_{role}.mp4"
        if raw.get("filename") != expected_name:
            return None, None
        candidate = entry.video_path(role)
        try:
            info = candidate.lstat()
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(entry.directory.resolve(strict=True))
        except (OSError, ValueError):
            return None, None
        if candidate.is_symlink() or not candidate.is_file() or info.st_size <= 0:
            return None, None
        camera_id = _safe_text(raw.get("camera_id"), 200)
        return resolved, camera_id

    def _recording_videos(
        self,
        path: Path,
        episode_id: str,
        entry: CacheEntry,
        inspection: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        shard = None
        if self._published_shard_provider is not None:
            try:
                shard = self._published_shard_provider(path)
            except (OSError, ValueError):
                shard = None
        result: dict[str, dict[str, Any]] = {}
        for role in VIDEO_FEATURES:
            video, feature = (
                self._shard_video(shard, role)
                if shard is not None
                else (None, None)
            )
            source = "lerobot" if video is not None else None
            if video is None:
                video, feature = self._cached_video(entry, inspection, role)
                if video is not None:
                    try:
                        data_format = normalize_data_format(
                            inspection.get("data_format")
                        )
                    except ValueError:
                        data_format = NEDF3_DATA_FORMAT
                    source = f"{data_format.casefold()}_cache"
            result[role] = {
                "role": role,
                "feature": feature,
                "source": source,
                "available": video is not None,
                "url": (
                    f"/api/recordings/{episode_id}/videos/{role}"
                    if video is not None
                    else None
                ),
                "size_bytes": video.stat().st_size if video is not None else 0,
            }
        return result

    def video_path(self, episode_id: str, role: str) -> Path:
        if role not in VIDEO_FEATURES:
            raise RecordingReplayRequestError("Invalid recording video role")
        path = self._episode(episode_id)
        shard = (
            self._published_shard_provider(path)
            if self._published_shard_provider is not None
            else None
        )
        video, _feature = (
            self._shard_video(shard, role)
            if shard is not None
            else (None, None)
        )
        if video is None:
            inspection, entry, _camera_roles = self._snapshot(path, episode_id)
            video, _feature = self._cached_video(entry, inspection, role)
        if video is None:
            raise RecordingReplayNotFound("Recorded video is not available")
        return video

    def summary(self, episode_id: str) -> dict[str, Any]:
        path = self._episode(episode_id)
        base = self._episode_metadata(path)
        date_name, episode_name = _episode_id_parts(episode_id) or (None, path.name)
        base["id"] = episode_id
        base["name"] = _episode_display_name(date_name, episode_name)
        _, segments, dense_issues = _dense_segments(path)
        checkpoints = [
            {
                "revision": segment.get("checkpoint_revision"),
                "frame": segment["end_frame"],
                "timestamp_ns": segment.get("end_timestamp_ns"),
                "label": segment["label"],
            }
            for segment in segments
        ]
        inspection, entry, camera_roles = self._snapshot(path, episode_id)
        issues = inspection.get("issues") if isinstance(inspection.get("issues"), list) else []
        return {
            **base,
            "cache_state": "ready",
            "data_format": inspection.get("data_format") or base.get("data_format"),
            "time_range_ns": inspection.get("time_range_ns"),
            "topics": inspection.get("topics", []),
            "cameras": inspection.get("cameras", []),
            "camera_roles": self._camera_roles(inspection, camera_roles),
            "joints": inspection.get("joints", {}),
            "tcp": inspection.get("tcp", {}),
            "grippers": inspection.get("grippers", {}),
            "gripper_curve": inspection.get("gripper_curve", {}),
            "videos": self._recording_videos(
                path,
                episode_id,
                entry,
                inspection,
            ),
            "capabilities": inspection.get("capabilities", {}),
            "dense": {
                "available": (path / "dense_labels.jsonl").is_file(),
                "segments": segments,
                "checkpoints": checkpoints,
            },
            "issues": issues + dense_issues,
        }

    @staticmethod
    def _nearest_timeline_values(
        timestamps: Any,
        values: Any,
        targets: Any,
    ) -> Any:
        import numpy as np

        indexes = np.searchsorted(timestamps, targets, side="left")
        indexes = np.clip(indexes, 0, len(timestamps) - 1)
        previous = np.maximum(indexes - 1, 0)
        use_previous = np.abs(targets - timestamps[previous]) <= np.abs(
            timestamps[indexes] - targets
        )
        indexes = np.where(use_previous, previous, indexes)
        return values[indexes]

    def playback_trajectory(
        self,
        episode_id: str,
        *,
        start: int = 0,
        end: Optional[int] = None,
        stride: int = 1,
        max_points: int = MAX_TRAJECTORY_POINTS,
    ) -> dict[str, Any]:
        for name, value, minimum, maximum in (
            ("start", start, 0, 10**9),
            ("stride", stride, 1, 10000),
            ("max_points", max_points, 1, MAX_TRAJECTORY_POINTS),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise RecordingReplayRequestError(
                    f"{name} is outside its allowed range"
                )
        if end is not None and (
            isinstance(end, bool)
            or not isinstance(end, int)
            or end < start
            or end > 10**9
        ):
            raise RecordingReplayRequestError("end is outside its allowed range")

        path = self._episode(episode_id)
        corruptions = 0
        while True:
            inspection, entry, _camera_roles = self._snapshot(path, episode_id)
            groups = {
                "tcp": inspection.get("tcp"),
                "joints": inspection.get("joints"),
                "grippers": inspection.get("grippers"),
            }
            groups = {
                name: value if isinstance(value, Mapping) else {}
                for name, value in groups.items()
            }
            topics = inspection.get("capabilities", {}).get("topics", {})
            if not isinstance(topics, Mapping):
                topics = {}

            stream_specs: dict[str, dict[str, Mapping[str, Any]]] = {}
            for group_name, group in groups.items():
                sides: dict[str, Mapping[str, Any]] = {}
                for side in ("left", "right"):
                    info = group.get(side)
                    if not isinstance(info, Mapping) or not info.get("available"):
                        continue
                    capability = topics.get(info.get("topic"))
                    if not isinstance(capability, Mapping):
                        continue
                    if not all(
                        isinstance(capability.get(key), str)
                        and capability.get(key)
                        for key in ("timestamp_key", "value_key")
                    ):
                        continue
                    sides[side] = capability
                stream_specs[group_name] = sides

            master = next(
                (
                    stream_specs[group_name][side]
                    for group_name, side in (
                        ("tcp", "left"),
                        ("tcp", "right"),
                        ("joints", "left"),
                        ("joints", "right"),
                    )
                    if side in stream_specs.get(group_name, {})
                ),
                None,
            )
            if master is None:
                raise RecordingReplayUnavailable(
                    "Recorded TCP and joint trajectories are unavailable",
                    availability={name: dict(value) for name, value in groups.items()},
                    code="recording_trajectory_unavailable",
                )

            cache_corrupt = False
            with self._cache.entry_lock(entry, timeout_s=DEFAULT_BUILD_WAIT_S) as acquired:
                if not acquired:
                    raise RecordingReplayBuilding(episode_id)
                entry_load = self._cache.load(entry, verify_hash=False)
                if entry_load.status == "corrupt":
                    self._cache.quarantine_locked(
                        entry, reason="playback_manifest_invalid"
                    )
                    cache_corrupt = True
                elif entry_load.status != "ready":
                    raise RecordingReplayBuilding(episode_id)
                try:
                    if cache_corrupt:
                        raise KeyError("playback_generation_quarantined")
                    import numpy as np

                    with np.load(entry.timeline_path, allow_pickle=False) as timeline:
                        master_timestamps = np.asarray(
                            timeline[master["timestamp_key"]], dtype=np.int64
                        )
                        loaded: dict[str, dict[str, tuple[Any, Any]]] = {}
                        stream_dimensions = {
                            "joints": 7,
                            "tcp": 7,
                            "grippers": 1,
                        }
                        for group_name, sides in stream_specs.items():
                            loaded[group_name] = {}
                            for side, capability in sides.items():
                                timestamps = np.asarray(
                                    timeline[capability["timestamp_key"]],
                                    dtype=np.int64,
                                )
                                values = np.asarray(
                                    timeline[capability["value_key"]],
                                    dtype=np.float32,
                                )
                                if (
                                    timestamps.ndim != 1
                                    or values.ndim != 2
                                    or values.shape
                                    != (
                                        timestamps.size,
                                        stream_dimensions[group_name],
                                    )
                                    or timestamps.size == 0
                                ):
                                    raise ValueError("invalid_playback_cache_shape")
                                loaded[group_name][side] = (timestamps, values)
                        if master_timestamps.ndim != 1 or master_timestamps.size == 0:
                            raise ValueError("invalid_playback_master_timeline")
                except (OSError, ValueError, KeyError):
                    if not cache_corrupt:
                        self._cache.quarantine_locked(
                            entry, reason="playback_load_failed"
                        )
                    cache_corrupt = True
            if cache_corrupt:
                corruptions += 1
                if corruptions > 2:
                    raise RecordingReplayBuildFailed(
                        episode_id,
                        "Recorded playback cache could not be rebuilt",
                        1000,
                    )
                continue

            total = int(master_timestamps.size)
            stop = total if end is None else min(end, total)
            indices = np.arange(min(start, total), stop, stride, dtype=np.int64)
            if int(indices.size) > max_points:
                indices = indices[
                    :: math.ceil(int(indices.size) / max_points)
                ]
            targets = master_timestamps[indices]
            payload_groups: dict[str, dict[str, Any]] = {}
            for group_name in ("joints", "tcp", "grippers"):
                sides = loaded.get(group_name, {})
                payload: dict[str, Any] = {
                    "available": bool(sides),
                    "left_available": "left" in sides,
                    "right_available": "right" in sides,
                    "left": [],
                    "right": [],
                }
                for side, (timestamps, values) in sides.items():
                    payload[side] = self._nearest_timeline_values(
                        timestamps, values, targets
                    ).tolist()
                payload_groups[group_name] = payload
            return {
                "id": episode_id,
                "cache_state": "ready",
                "schema_version": 3,
                "start": start,
                "end": stop,
                "stride": stride,
                "total_frames": total,
                "returned_frames": int(indices.size),
                "timestamps_ns": targets.tolist(),
                **payload_groups,
                "issues": [],
            }

    def trajectory(
        self,
        episode_id: str,
        *,
        start: int = 0,
        end: Optional[int] = None,
        stride: int = 1,
        max_points: int = MAX_TRAJECTORY_POINTS,
    ) -> dict[str, Any]:
        for name, value, minimum, maximum in (
            ("start", start, 0, 10**9),
            ("stride", stride, 1, 10000),
            ("max_points", max_points, 1, MAX_TRAJECTORY_POINTS),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise RecordingReplayRequestError(f"{name} is outside its allowed range")
        if end is not None and (isinstance(end, bool) or not isinstance(end, int) or end < start or end > 10**9):
            raise RecordingReplayRequestError("end is outside its allowed range")
        path = self._episode(episode_id)
        corruptions = 0
        while True:
            inspection, entry, _camera_roles = self._snapshot(path, episode_id)
            joints = inspection.get("joints") if isinstance(inspection.get("joints"), Mapping) else {}
            left_info = joints.get("left") if isinstance(joints.get("left"), Mapping) else {}
            right_info = joints.get("right") if isinstance(joints.get("right"), Mapping) else {}
            availability = {"left": dict(left_info), "right": dict(right_info)}
            if not left_info.get("available") or not right_info.get("available"):
                raise RecordingReplayUnavailable(
                    "Recorded joint positions are unavailable; TCP poses are not converted with IK",
                    availability=availability,
                )
            capabilities = inspection.get("capabilities") if isinstance(inspection.get("capabilities"), Mapping) else {}
            topics = capabilities.get("topics") if isinstance(capabilities.get("topics"), Mapping) else {}
            left_capability = topics.get(left_info.get("topic")) if isinstance(topics.get(left_info.get("topic")), Mapping) else {}
            right_capability = topics.get(right_info.get("topic")) if isinstance(topics.get(right_info.get("topic")), Mapping) else {}
            required_keys = (
                left_capability.get("timestamp_key"),
                left_capability.get("value_key"),
                right_capability.get("timestamp_key"),
                right_capability.get("value_key"),
            )
            if not all(isinstance(value, str) and value for value in required_keys):
                raise RecordingReplayUnavailable("Recorded joint trajectory cache is incomplete", availability=availability)
            cache_corrupt = False
            with self._cache.entry_lock(entry, timeout_s=DEFAULT_BUILD_WAIT_S) as acquired:
                if not acquired:
                    raise RecordingReplayBuilding(episode_id)
                entry_load = self._cache.load(entry, verify_hash=False)
                if entry_load.status == "corrupt":
                    self._cache.quarantine_locked(entry, reason="trajectory_manifest_invalid")
                    cache_corrupt = True
                elif entry_load.status != "ready":
                    raise RecordingReplayBuilding(episode_id)
                try:
                    if cache_corrupt:
                        raise KeyError("trajectory_generation_quarantined")
                    import numpy as np

                    with np.load(entry.timeline_path, allow_pickle=False) as timeline:
                        left_timestamps = np.asarray(timeline[required_keys[0]], dtype=np.int64)
                        left_values = np.asarray(timeline[required_keys[1]], dtype=np.float32)
                        right_timestamps = np.asarray(timeline[required_keys[2]], dtype=np.int64)
                        right_values = np.asarray(timeline[required_keys[3]], dtype=np.float32)
                    valid_shape = (
                        left_values.ndim == 2
                        and right_values.ndim == 2
                        and left_values.shape[1:] == (7,)
                        and right_values.shape[1:] == (7,)
                        and left_timestamps.ndim == 1
                        and right_timestamps.ndim == 1
                        and left_values.shape[0] == left_timestamps.shape[0]
                        and right_values.shape[0] == right_timestamps.shape[0]
                        and np.array_equal(left_timestamps, right_timestamps)
                    )
                    if not valid_shape:
                        raise ValueError("invalid_joint_cache_shape")
                except (OSError, ValueError, KeyError):
                    if not cache_corrupt:
                        self._cache.quarantine_locked(entry, reason="trajectory_load_failed")
                    cache_corrupt = True
            if cache_corrupt:
                corruptions += 1
                if corruptions > 2:
                    raise RecordingReplayBuildFailed(
                        episode_id,
                        "Recorded joint trajectory cache could not be rebuilt",
                        1000,
                    )
                # The next snapshot call enqueues one deduplicated rebuild.  It
                # returns 202/503 through the normal asynchronous contract.
                continue
            total = int(left_timestamps.shape[0])
            stop = total if end is None else min(end, total)
            indices = np.arange(min(start, total), stop, stride, dtype=np.int64)
            if int(indices.size) > max_points:
                indices = indices[:: math.ceil(int(indices.size) / max_points)]
            return {
                "id": episode_id,
                "cache_state": "ready",
                "schema_version": 1,
                "start": start,
                "end": stop,
                "stride": stride,
                "total_frames": total,
                "returned_frames": int(indices.size),
                "timestamps_ns": left_timestamps[indices].tolist(),
                "left": left_values[indices].tolist(),
                "right": right_values[indices].tolist(),
                "issues": [],
            }
