#!/usr/bin/env python3
"""
Episode-level upload queue for Vive/DM3 collection data.

The manager owns the single upload path used by both automatic and manual
confirmation flows. It scans finalized NEDF3 episode folders, applies the
operator safety policy, uploads one episode at a time, persists per-episode
state, and samples machine performance for warning display.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

import psutil

from utils.logger_config import logger
from incubator.nedf3.Vive_DM3.http_upload_backend import upload_episode_to_http_sink
from incubator.nedf3.Vive_DM3.resource_policy import apply_upload_thread_policy
from incubator.nedf3.windows_DM3.oss_upload_config import OssMultipartUploadConfig
from incubator.nedf3.windows_DM3.oss_uploader import UploadCancelled, UploadProgress, UploadResult, upload_episode_data_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERF_LOG_PATH = PROJECT_ROOT / "logs" / "upload_manager_perf.jsonl"
PERF_LOG_MAX_BYTES = 128 * 1024 * 1024
PERF_EVENT_WARN_SEC = 0.10
PERF_EPISODE_REFRESH_WARN_SEC = 0.05
PERF_DISCOVERY_WARN_SEC = 0.05
PERF_EVENT_MIN_INTERVAL_SEC = 30.0
PERF_SAMPLE_LOG_INTERVAL_SEC = 10.0
STATE_FILE_NAME = "auto_upload_state.json"
DEFAULT_RESOURCE_TYPE = "robopocket20_pro"
DEFAULT_PROTECT_MINUTES = 10
DEFAULT_KEEP_LATEST_COUNT = 5
DEFAULT_POLL_INTERVAL_SEC = 0
DEFAULT_MAX_RETRIES = 5
DEFAULT_UPLOAD_WORKERS = 2
DEFAULT_MULTIPART_THREADS = 4
SNAPSHOT_INCREMENTAL_REFRESH_SEC = 5.0
SNAPSHOT_INCREMENTAL_BATCH_SIZE = 4
SNAPSHOT_DISCOVERY_INTERVAL_SEC = 10.0
UPLOAD_STATUS_FILE_NAME = ".auto_upload_status.json"
UPLOAD_STATUS_WRITE_INTERVAL_SEC = 1.0
DEFAULT_UPLOAD_STATUS_PATH = PROJECT_ROOT.parent / "data" / UPLOAD_STATUS_FILE_NAME
UPLOAD_BACKEND_OSS = "oss"
UPLOAD_BACKEND_HTTP_SINK = "http_sink"


LogCallback = Callable[[str], None]
TokenFetcher = Callable[[str], Dict[str, Any]]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_error(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__


def append_perf_log(snapshot: Dict[str, Any], log_path: Optional[Path] = DEFAULT_PERF_LOG_PATH) -> None:
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists() and log_path.stat().st_size >= PERF_LOG_MAX_BYTES:
            rotated = log_path.with_suffix(log_path.suffix + ".1")
            if rotated.exists():
                rotated.unlink()
            log_path.replace(rotated)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception as exc:
        logger.debug(f"Failed to append performance log: {exc}")


def episode_key(path: Path) -> str:
    return os.path.normcase(str(path.expanduser().resolve()))


@dataclass(frozen=True)
class EpisodeInfo:
    name: str
    path: Path
    mtime: float
    size_bytes: int
    file_count: int
    complete: bool
    upload_record_status: str = ""
    upload_record_note: str = ""
    details_loaded: bool = True

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.mtime)

    @property
    def id(self) -> str:
        return episode_key(self.path)


@dataclass(frozen=True)
class UploadPolicy:
    protect_minutes: int = DEFAULT_PROTECT_MINUTES
    keep_latest_count: int = DEFAULT_KEEP_LATEST_COUNT


@dataclass(frozen=True)
class UploadJob:
    episode_id: str
    max_attempts: Optional[int] = None


@dataclass(frozen=True)
class EpisodeMaintenanceResult:
    episodes: List[EpisodeInfo]
    scan_mode: str
    target_found: bool


class PerformanceMonitor:
    """Lightweight machine-level performance sampler for upload warnings."""

    def __init__(
        self,
        *,
        cpu_warn_percent: float = 85.0,
        memory_warn_percent: float = 90.0,
        disk_write_warn_mb_s: float = 120.0,
        sample_interval_sec: float = 5.0,
        log_path: Optional[Path] = DEFAULT_PERF_LOG_PATH,
        source: str = "upload_manager",
    ) -> None:
        self.cpu_warn_percent = cpu_warn_percent
        self.memory_warn_percent = memory_warn_percent
        self.disk_write_warn_mb_s = disk_write_warn_mb_s
        self.sample_interval_sec = sample_interval_sec
        self.log_path = log_path
        self.source = source
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_disk = None
        self._last_net = None
        self._last_time = 0.0
        self._last_log_time = 0.0
        self._snapshot: Dict[str, Any] = {
            "running": False,
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "disk_read_mb_s": 0.0,
            "disk_write_mb_s": 0.0,
            "net_upload_mb_s": 0.0,
            "net_download_mb_s": 0.0,
            "alerts": [],
            "updated_at": "",
        }

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        psutil.cpu_percent(interval=None)
        self._last_disk = psutil.disk_io_counters()
        self._last_net = psutil.net_io_counters()
        self._last_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        with self._lock:
            self._snapshot["running"] = False

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._snapshot)

    def _run(self) -> None:
        apply_upload_thread_policy()
        while not self._stop_event.wait(self.sample_interval_sec):
            self._sample_once()

    def _sample_once(self) -> None:
        now = time.time()
        disk = psutil.disk_io_counters()
        net = psutil.net_io_counters()
        elapsed = max(0.001, now - self._last_time)

        disk_read_mb_s = 0.0
        disk_write_mb_s = 0.0
        if self._last_disk is not None and disk is not None:
            disk_read_mb_s = max(0.0, disk.read_bytes - self._last_disk.read_bytes) / elapsed / (1024**2)
            disk_write_mb_s = max(0.0, disk.write_bytes - self._last_disk.write_bytes) / elapsed / (1024**2)

        net_upload_mb_s = 0.0
        net_download_mb_s = 0.0
        if self._last_net is not None and net is not None:
            net_upload_mb_s = max(0.0, net.bytes_sent - self._last_net.bytes_sent) / elapsed / (1024**2)
            net_download_mb_s = max(0.0, net.bytes_recv - self._last_net.bytes_recv) / elapsed / (1024**2)

        cpu_percent = float(psutil.cpu_percent(interval=None))
        memory_percent = float(psutil.virtual_memory().percent)

        alerts: List[str] = []
        if cpu_percent >= self.cpu_warn_percent:
            alerts.append(f"CPU high: {cpu_percent:.0f}%")
        if memory_percent >= self.memory_warn_percent:
            alerts.append(f"Memory high: {memory_percent:.0f}%")
        if disk_write_mb_s >= self.disk_write_warn_mb_s:
            alerts.append(f"Disk write high: {disk_write_mb_s:.1f} MB/s")

        with self._lock:
            snapshot = {
                "running": True,
                "source": self.source,
                "pid": os.getpid(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_read_mb_s": disk_read_mb_s,
                "disk_write_mb_s": disk_write_mb_s,
                "net_upload_mb_s": net_upload_mb_s,
                "net_download_mb_s": net_download_mb_s,
                "alerts": alerts,
                "updated_at": utc_now_iso(),
            }
            self._snapshot = snapshot

        self._last_disk = disk
        self._last_net = net
        self._last_time = now
        if alerts or now - self._last_log_time >= PERF_SAMPLE_LOG_INTERVAL_SEC:
            self._append_perf_log(snapshot)
            self._last_log_time = now

    def _append_perf_log(self, snapshot: Dict[str, Any]) -> None:
        append_perf_log(snapshot, self.log_path)


class AutoUploadManager:
    """Single upload engine for auto, force-all, and retry flows."""

    def __init__(
        self,
        *,
        data_dir: Path,
        data_dirs: Optional[Iterable[Path]] = None,
        token_fetcher: TokenFetcher,
        log_callback: Optional[LogCallback] = None,
    ) -> None:
        self.data_dir = data_dir.expanduser().resolve()
        self.data_dirs: List[Path] = []
        self._set_scan_dirs([self.data_dir, *(data_dirs or [])])
        self.token_fetcher = token_fetcher
        self.log_callback = log_callback

        self.resource_type = DEFAULT_RESOURCE_TYPE
        self.policy = UploadPolicy()
        self.poll_interval_sec = DEFAULT_POLL_INTERVAL_SEC
        self.max_retries = DEFAULT_MAX_RETRIES
        self.upload_workers = DEFAULT_UPLOAD_WORKERS
        self.multipart_threads = DEFAULT_MULTIPART_THREADS
        self.upload_backend = UPLOAD_BACKEND_OSS
        self.http_sink_url = ""
        self.http_sink_token = ""
        self.http_timeout_sec = 120

        self._state_lock = threading.RLock()
        self._scan_lock = threading.RLock()
        self._queue: deque[UploadJob] = deque()
        self._queue_cond = threading.Condition(self._state_lock)
        self._queued: Set[str] = set()
        self._current_episode: Optional[str] = None
        self._current_progress: Optional[UploadProgress] = None
        self._stop_event = threading.Event()
        self._auto_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._auto_thread: Optional[threading.Thread] = None
        self._state: Dict[str, Any] = {"version": 1, "episodes": {}}
        self._episode_cache: Dict[str, EpisodeInfo] = {}
        self._episode_cache_order: List[str] = []
        self._episode_cache_refreshed_at: Dict[str, float] = {}
        self._episode_cache_initialized = False
        self._snapshot_refresh_index = 0
        self._last_snapshot_discovery = 0.0
        self._last_perf_event_at: Dict[str, float] = {}
        self._last_status_write_at = 0.0
        self._background_refresh_enabled = True
        self._auto_enqueue_suspended = False

        # Keep filesystem I/O out of _state_lock. A single persistence thread
        # coalesces revisions so older snapshots cannot overwrite newer state.
        self._persistence_cond = threading.Condition(self._state_lock)
        self._persistence_event = threading.Event()
        self._persistence_stop_event = threading.Event()
        self._state_revision = 0
        self._persisted_state_revision = 0
        self._status_revision = 0
        self._persisted_status_revision = 0
        self._pending_status_payload: Optional[Dict[str, Any]] = None
        self._pending_status_paths: List[Path] = []
        self._persistence_thread = threading.Thread(
            target=self._persistence_loop,
            name="dm3-upload-state-writer",
            daemon=True,
        )

        self.performance = PerformanceMonitor()
        self._load_state()
        self._persistence_thread.start()
        self.performance.start()

    def configure(
        self,
        *,
        data_dir: Optional[Path] = None,
        data_dirs: Optional[Iterable[Path]] = None,
        resource_type: Optional[str] = None,
        protect_minutes: Optional[int] = None,
        keep_latest_count: Optional[int] = None,
        poll_interval_sec: Optional[int] = None,
        max_retries: Optional[int] = None,
        upload_workers: Optional[int] = None,
        multipart_threads: Optional[int] = None,
        upload_backend: Optional[str] = None,
        http_sink_url: Optional[str] = None,
        http_sink_token: Optional[str] = None,
        http_timeout_sec: Optional[int] = None,
    ) -> None:
        with self._state_lock:
            if data_dir is not None:
                resolved_dir = data_dir.expanduser().resolve()
                if resolved_dir != self.data_dir:
                    self.data_dir = resolved_dir
                    self._set_scan_dirs([*self.data_dirs, resolved_dir])
                    self._reset_episode_cache_locked()
            if data_dirs is not None:
                self._set_scan_dirs([*self.data_dirs, *data_dirs])
                self._reset_episode_cache_locked()
            if resource_type is not None:
                value = resource_type.strip()
                self.resource_type = value
            if protect_minutes is not None:
                self.policy = UploadPolicy(
                    protect_minutes=max(0, int(protect_minutes)),
                    keep_latest_count=self.policy.keep_latest_count,
                )
            if keep_latest_count is not None:
                self.policy = UploadPolicy(
                    protect_minutes=self.policy.protect_minutes,
                    keep_latest_count=max(0, int(keep_latest_count)),
                )
            if poll_interval_sec is not None:
                self.poll_interval_sec = max(0, int(poll_interval_sec))
            if max_retries is not None:
                self.max_retries = max(0, int(max_retries))
            if upload_workers is not None:
                self.upload_workers = max(1, int(upload_workers))
            if multipart_threads is not None:
                self.multipart_threads = max(1, int(multipart_threads))
            if upload_backend is not None:
                backend = upload_backend.strip().lower()
                if backend not in {UPLOAD_BACKEND_OSS, UPLOAD_BACKEND_HTTP_SINK}:
                    raise ValueError(f"Unsupported upload backend: {upload_backend}")
                self.upload_backend = backend
            if http_sink_url is not None:
                self.http_sink_url = http_sink_url.strip()
            if http_sink_token is not None:
                self.http_sink_token = http_sink_token.strip()
            if http_timeout_sec is not None:
                self.http_timeout_sec = max(5, int(http_timeout_sec))
            self._state["resource_type"] = self.resource_type
            self._state["data_dir"] = str(self.data_dir)
            self._state["data_dirs"] = [str(path) for path in self.data_dirs]
            self._state["policy"] = {
                "protect_minutes": self.policy.protect_minutes,
                "keep_latest_count": self.policy.keep_latest_count,
                "maintenance_trigger": "after_episode",
                "poll_interval_sec": self.poll_interval_sec,
                "max_retries": self.max_retries,
                "upload_workers": self.upload_workers,
                "multipart_threads": self.multipart_threads,
                "upload_backend": self.upload_backend,
                "http_sink_url": self.http_sink_url,
                "http_timeout_sec": self.http_timeout_sec,
            }
            self._save_state_locked()

    def start_worker(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def start_auto(self) -> None:
        self.start_worker()
        was_enabled = self._auto_event.is_set()
        if self._auto_thread is not None and self._auto_thread.is_alive():
            if not was_enabled:
                self._auto_event.set()
                self._log("自动上传已启动")
            return
        self._auto_event.set()
        self._auto_thread = threading.Thread(target=self._auto_loop, daemon=True)
        self._auto_thread.start()
        self._log("自动上传已启动")

    def stop_auto(self) -> None:
        if not self._auto_event.is_set():
            return
        self._auto_event.clear()
        self._log("自动上传已停止")

    def set_background_refresh_enabled(self, enabled: bool) -> None:
        with self._state_lock:
            self._background_refresh_enabled = bool(enabled)

    def set_auto_enqueue_suspended(self, suspended: bool) -> None:
        with self._state_lock:
            self._auto_enqueue_suspended = bool(suspended)

    def shutdown(self) -> None:
        if self._persistence_stop_event.is_set():
            return
        self._auto_event.clear()
        self._stop_event.set()
        with self._queue_cond:
            self._queue_cond.notify_all()
        if self._auto_thread is not None:
            self._auto_thread.join(timeout=3.0)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
        self.performance.stop()
        with self._state_lock:
            self._mark_interrupted_transient_states_locked()
            self._save_state_locked()
            self._current_episode = None
            self._current_progress = None
            self._write_status_snapshot_locked(force=True)
        if not self.flush_persistence(timeout=5.0):
            logger.warning("Timed out waiting for upload state persistence during shutdown")
        self._persistence_stop_event.set()
        self._persistence_event.set()
        self._persistence_thread.join(timeout=5.0)

    def scan(self) -> List[EpisodeInfo]:
        with self._scan_lock:
            episodes = scan_episode_dirs_many(self.data_dirs)
            self._replace_episode_cache(episodes)
            return episodes

    def refresh_after_episode(self, episode_path: Path) -> EpisodeMaintenanceResult:
        """Refresh one finalized episode, falling back to one full cache initialization."""
        target = self._validate_managed_episode_path(episode_path)
        with self._scan_lock:
            with self._state_lock:
                initialized = self._episode_cache_initialized

            if not initialized:
                episodes = scan_episode_dirs_many(self.data_dirs)
                self._replace_episode_cache(episodes)
                return EpisodeMaintenanceResult(
                    episodes=episodes,
                    scan_mode="full_init",
                    target_found=any(info.id == episode_key(target) for info in episodes),
                )

            refreshed = scan_episode_dir(target)
            if refreshed is None:
                self._forget_missing_episode(episode_key(target))
            else:
                self._cache_episode(refreshed)
                if refreshed.complete and not is_complete_episode_dir(target):
                    self._forget_missing_episode(refreshed.id)
                    refreshed = None

            episodes = self._cached_episodes()
            episodes = self._prune_missing_protection_entries(episodes)
            return EpisodeMaintenanceResult(
                episodes=episodes,
                scan_mode="incremental",
                target_found=refreshed is not None,
            )

    def enqueue_eligible(self, *, force: bool = False) -> int:
        self.start_worker()
        with self._state_lock:
            background_refresh = self._background_refresh_enabled
        episodes = self.scan() if force else self._snapshot_episodes(background_refresh=background_refresh)
        return self.enqueue_eligible_episodes(episodes, force=force)

    def enqueue_eligible_episodes(
        self,
        episodes: List[EpisodeInfo],
        *,
        force: bool = False,
        timings: Optional[Dict[str, float]] = None,
    ) -> int:
        self.start_worker()
        phase_started = time.perf_counter()
        episodes = self._prune_missing_protection_entries(episodes)
        if timings is not None:
            timings["prune_sec"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        eligible = self._eligible_episode_ids(episodes, force=force)
        if timings is not None:
            timings["eligibility_sec"] = time.perf_counter() - phase_started
        episode_by_id = {episode.id: episode for episode in episodes}
        ready_ids: List[str] = []
        phase_started = time.perf_counter()
        for episode_id in eligible:
            episode = episode_by_id.get(episode_id)
            if episode is None or not is_complete_episode_dir(episode.path):
                self._forget_missing_episode(episode_id)
                continue
            ready_ids.append(episode_id)
        if timings is not None:
            timings["validation_sec"] = time.perf_counter() - phase_started
        count = self._enqueue_episode_ids(ready_ids, timings=timings)
        if count:
            self._log(f"Queued {count} episode(s)")
        return count

    def enqueue_failed(self) -> int:
        with self._scan_lock, self._state_lock:
            persisted = dict(self._state.get("episodes", {}))
        failed_paths = [Path(row["path"]) for row in self._failed_file_tree(self.scan(), persisted)]
        count = self.force_enqueue_paths(failed_paths)
        if count:
            self._log(f"Force queued {count} failed episode(s) for retry")
        return count

    def force_enqueue_path(self, path: Path) -> int:
        return self.force_enqueue_paths([path])

    def force_enqueue_paths(self, paths: Iterable[Path]) -> int:
        """Queue complete episodes under a path at the front for one upload attempt."""
        self.start_worker()
        targets = [path.expanduser().resolve() for path in paths]
        with self._state_lock:
            states = dict(self._state.get("episodes", {}))
        episodes = self.scan()
        name_counts = self._episode_name_counts(episodes)
        selected: List[EpisodeInfo] = []
        for episode in episodes:
            if not episode.complete:
                continue
            if not self._failure_reason(episode, states.get(episode.id, self._legacy_state_for(episode, name_counts))):
                continue
            matches = False
            for target in targets:
                try:
                    episode_path = episode.path.resolve()
                    matches = (
                        episode_path == target
                        or episode_path.is_relative_to(target)
                        or target.is_relative_to(episode_path)
                    )
                except OSError:
                    matches = False
                if matches:
                    break
            if matches:
                selected.append(episode)
        selected.sort(key=lambda item: item.mtime)
        seen: Set[str] = set()
        unique_selected: List[EpisodeInfo] = []
        for episode in selected:
            if episode.id in seen:
                continue
            seen.add(episode.id)
            unique_selected.append(episode)
        count = self._enqueue_episode_ids(
            (episode.id for episode in reversed(unique_selected)),
            front=True,
            max_attempts=1,
        )
        if count:
            self._log(f"Force queued {count} episode(s) at front")
        return count

    def refresh_snapshot_cache(self, *, full: bool = False, background_refresh: bool = True) -> None:
        """Refresh cached episode metadata without forcing every UI tick to scan all files."""
        with self._state_lock:
            current = self._current_episode
        if full:
            self.scan()
            return

        now = time.time()
        if background_refresh and (
            not self._episode_cache or now - self._last_snapshot_discovery >= SNAPSHOT_DISCOVERY_INTERVAL_SEC
        ):
            self._discover_snapshot_episode_dirs()
            self._last_snapshot_discovery = now

        if current and current not in self._episode_cache:
            self._refresh_cached_episode(current)

        if not background_refresh:
            return

        refresh_ids = self._next_snapshot_refresh_ids(current)
        for episode_id in refresh_ids:
            cached = self._episode_cache.get(episode_id)
            if cached is None:
                continue
            if now - self._episode_cache_refreshed_at.get(episode_id, 0.0) < SNAPSHOT_INCREMENTAL_REFRESH_SEC:
                continue
            self._refresh_cached_episode(episode_id)

    def snapshot(self, *, background_refresh: bool = True) -> Dict[str, Any]:
        started_at = time.perf_counter()
        episodes = self._snapshot_episodes(background_refresh=background_refresh)
        complete_count = sum(1 for item in episodes if item.complete)
        protected = self._protected_reasons(episodes)
        name_counts = self._episode_name_counts(episodes)

        with self._state_lock:
            persisted = dict(self._state.get("episodes", {}))
            queued = list(self._queued)
            queue_order = [job.episode_id for job in self._queue]
            current = self._current_episode
            progress = self._progress_dict(self._current_progress)

        rows: List[Dict[str, Any]] = []
        for info in episodes:
            episode_id = info.id
            state = dict(persisted.get(episode_id, self._legacy_state_for(info, name_counts)))
            status = state.get("status") or "pending"
            failure_reason = self._failure_reason(info, state)
            if episode_id in queued and status not in {"uploading", "success"}:
                status = "queued"
            elif episode_id == current:
                status = "uploading"
            elif failure_reason and status != "success":
                status = "failed"
            elif status == "pending" and info.upload_record_status == "success":
                status = "success"
            elif not info.complete and info.details_loaded:
                status = "incomplete"
            elif status == "pending" and episode_id in protected:
                status = "protected"

            rows.append(
                {
                    "id": episode_id,
                    "name": info.name,
                    "task_name": info.path.parent.name,
                    "path": str(info.path),
                    "mtime": info.mtime,
                    "age_seconds": info.age_seconds,
                    "size_bytes": info.size_bytes,
                    "file_count": info.file_count,
                    "complete": info.complete,
                    "status": status,
                    "reason": failure_reason if status == "failed" else state.get("last_error") or protected.get(episode_id, ""),
                    "retry_count": int(state.get("retry_count") or 0),
                    "updated_at": state.get("updated_at", ""),
                    "progress": self._episode_progress(episode_id, status, progress),
                }
            )

        stats = {
            "total": len(rows),
            "complete": complete_count,
            "queued": len(queued),
            "success": sum(1 for row in rows if row["status"] == "success"),
            "failed": sum(1 for row in rows if row["status"] == "failed"),
            "protected": sum(1 for row in rows if row["status"] == "protected"),
        }
        queue_rows = self._queue_display_rows(rows, current, queue_order)
        failed_file_tree = self._failed_file_tree(episodes, persisted)
        snapshot = {
            "data_dir": str(self.data_dir),
            "data_dirs": [str(path) for path in self.data_dirs],
            "resource_type": self.resource_type,
            "auto_enabled": self._auto_event.is_set(),
            "worker_running": self._worker_thread is not None and self._worker_thread.is_alive(),
            "current_episode": current,
            "current_progress": progress,
            "policy": {
                "protect_minutes": self.policy.protect_minutes,
                "keep_latest_count": self.policy.keep_latest_count,
                "maintenance_trigger": "after_episode",
                "poll_interval_sec": self.poll_interval_sec,
                "max_retries": self.max_retries,
                "upload_workers": self.upload_workers,
                "multipart_threads": self.multipart_threads,
                "upload_backend": self.upload_backend,
                "http_sink_url": self.http_sink_url,
                "http_timeout_sec": self.http_timeout_sec,
            },
            "stats": stats,
            "episodes": rows,
            "queue_episodes": queue_rows,
            "failed_file_tree": failed_file_tree,
            "performance": self.performance.snapshot(),
        }
        elapsed_sec = time.perf_counter() - started_at
        if elapsed_sec >= PERF_EVENT_WARN_SEC:
            self._append_perf_event(
                "auto_upload_snapshot_slow",
                elapsed_sec=elapsed_sec,
                episode_count=len(episodes),
                cached_episode_count=len(episodes),
                queue_count=len(queued),
                current_episode=bool(current),
            )
        return snapshot

    def upload_progress_snapshot(self) -> Dict[str, Any]:
        """Return live upload progress without scanning or rebuilding episode rows."""
        with self._state_lock:
            current = self._current_episode
            progress = self._progress_dict(self._current_progress)
            queued_count = len(self._queued)
            queue_count = len(self._queue)
            auto_enabled = self._auto_event.is_set()
            worker_running = self._worker_thread is not None and self._worker_thread.is_alive()
            cached = self._episode_cache.get(current) if current else None
            state = dict(self._state.get("episodes", {}).get(current, {})) if current else {}
        return {
            "data_dir": str(self.data_dir),
            "auto_enabled": auto_enabled,
            "worker_running": worker_running,
            "current_episode": current,
            "episode_name": cached.name if cached is not None else (Path(current).name if current else ""),
            "episode_path": str(cached.path) if cached is not None else (current or ""),
            "progress": progress,
            "queued_count": queued_count,
            "queue_count": queue_count,
            "updated_at": utc_now_iso(),
            "status": state.get("status", "uploading" if current else ""),
            "performance": self.performance.snapshot(),
        }

    def _snapshot_episodes(self, *, background_refresh: bool = True) -> List[EpisodeInfo]:
        self.refresh_snapshot_cache(full=False, background_refresh=background_refresh)
        with self._state_lock:
            episodes = list(self._episode_cache.values())
        episodes.sort(key=lambda item: item.mtime)
        return episodes

    @staticmethod
    def _queue_display_rows(
        rows: List[Dict[str, Any]],
        current: Optional[str],
        queue_order: List[str],
    ) -> List[Dict[str, Any]]:
        order: Dict[str, int] = {}
        rank = 0
        if current:
            order[current] = rank
            rank += 1
        for name in queue_order:
            if name not in order:
                order[name] = rank
                rank += 1

        queue_rows = [row for row in rows if row["status"] not in {"failed", "success"}]
        original_index = {row["id"]: index for index, row in enumerate(queue_rows)}
        return sorted(
            queue_rows,
            key=lambda row: (order.get(row["id"], rank + original_index[row["id"]])),
        )

    def _auto_loop(self) -> None:
        apply_upload_thread_policy()
        while not self._stop_event.is_set():
            # Automatic discovery is intentionally event-driven by the saver
            # between episodes. This thread only keeps auto-upload state alive
            # without doing fixed-interval disk scans.
            if self._stop_event.wait(1.0):
                break

    def _worker_loop(self) -> None:
        apply_upload_thread_policy()
        while not self._stop_event.is_set():
            with self._queue_cond:
                while not self._queue and not self._stop_event.is_set():
                    self._queue_cond.wait(timeout=0.2)
                if self._stop_event.is_set():
                    break
                job = self._queue.popleft()
                episode_id = job.episode_id

            with self._state_lock:
                self._queued.discard(episode_id)
                self._current_episode = episode_id
                self._current_progress = None
                self._write_status_snapshot_locked(force=True)
            try:
                self._upload_episode_with_retries(episode_id, max_attempts=job.max_attempts)
            finally:
                with self._state_lock:
                    if self._current_episode == episode_id:
                        self._current_episode = None
                        self._current_progress = None
                    self._write_status_snapshot_locked(force=True)

    def _upload_episode_with_retries(self, episode_id: str, *, max_attempts: Optional[int] = None) -> None:
        attempt = 0
        allowed_attempts = max_attempts if max_attempts is not None else self.max_retries + 1
        while not self._stop_event.is_set():
            episode_path = self._episode_path_for_id(episode_id)
            episode_name = episode_path.name
            if not episode_path.is_dir():
                self._forget_missing_episode(episode_id)
                self._log(f"Skipped missing episode {episode_name}; it may have been deleted")
                return

            attempt += 1
            status = "uploading" if attempt == 1 else "retrying"
            self._set_episode_state(episode_id, status=status, last_error="")
            self._log(f"Uploading {episode_name} (attempt {attempt}/{allowed_attempts})")

            try:
                result = self._upload_episode(episode_path)
            except UploadCancelled as exc:
                self._set_episode_state(
                    episode_id,
                    status="pending",
                    attempt_count=attempt,
                    retry_count=max(0, attempt - 1),
                    last_error=_format_error(exc),
                )
                self._log(f"Upload cancelled: {episode_name}")
                return
            except Exception as exc:
                error = _format_error(exc)
                self._set_episode_state(
                    episode_id,
                    status="failed",
                    attempt_count=attempt,
                    retry_count=max(0, attempt - 1),
                    last_error=error,
                )
                if attempt >= allowed_attempts:
                    self._log(f"Upload failed after retries: {episode_name}: {error}")
                    return
                delay = min(300.0, 10.0 * (2 ** (attempt - 1)))
                self._set_episode_state(
                    episode_id,
                    status="retrying",
                    attempt_count=attempt,
                    retry_count=max(0, attempt - 1),
                    last_error=f"{error}; retry in {delay:.0f}s",
                )
                self._log(f"Upload error for {episode_name}: {error}; retry in {delay:.0f}s")
                if self._stop_event.wait(delay):
                    return
                continue

            self._set_episode_success(episode_id, result)
            self._log(f"Uploaded {episode_name}")
            return

    def _upload_episode(self, episode_path: Path) -> UploadResult:
        if self.upload_backend == UPLOAD_BACKEND_HTTP_SINK:
            return self._upload_episode_to_http_sink(episode_path)
        return self._upload_episode_to_oss(episode_path)

    def _upload_episode_to_oss(self, episode_path: Path) -> UploadResult:
        token = self.token_fetcher(self.resource_type)

        def refresh_token() -> Dict[str, Any]:
            self._log("STS token expired; refreshing")
            return self.token_fetcher(self.resource_type)

        def progress_message(message: str) -> None:
            self._log(message)

        def upload_progress(progress: UploadProgress) -> None:
            with self._state_lock:
                self._current_progress = progress
                if self._current_episode:
                    self._state_for(self._current_episode).update(
                        {
                            "completed_files": progress.completed_files,
                            "total_files": progress.total_files,
                            "completed_bytes": progress.completed_bytes,
                            "total_bytes": progress.total_bytes,
                            "current_bytes": progress.current_bytes,
                            "updated_at": utc_now_iso(),
                        }
                    )
                self._write_status_snapshot_locked(force=False)

        return upload_episode_data_dir(
            episode_path,
            token,
            progress=progress_message,
            refresh_sts_token=refresh_token,
            max_workers=self.upload_workers,
            on_upload_progress=upload_progress,
            multipart_config=OssMultipartUploadConfig(num_threads=self.multipart_threads),
            stop_event=self._stop_event,
        )

    def _upload_episode_to_http_sink(self, episode_path: Path) -> UploadResult:
        if not self.http_sink_url:
            raise ValueError("HTTP receiver URL is required for pressure-test upload")

        def progress_message(message: str) -> None:
            self._log(message)

        def upload_progress(progress: UploadProgress) -> None:
            with self._state_lock:
                self._current_progress = progress
                if self._current_episode:
                    self._state_for(self._current_episode).update(
                        {
                            "completed_files": progress.completed_files,
                            "total_files": progress.total_files,
                            "completed_bytes": progress.completed_bytes,
                            "total_bytes": progress.total_bytes,
                            "current_bytes": progress.current_bytes,
                            "updated_at": utc_now_iso(),
                        }
                    )
                self._write_status_snapshot_locked(force=False)

        return upload_episode_to_http_sink(
            episode_path,
            self.http_sink_url,
            progress=progress_message,
            max_workers=self.upload_workers,
            on_upload_progress=upload_progress,
            timeout_sec=float(self.http_timeout_sec),
            token=self.http_sink_token,
            stop_event=self._stop_event,
        )

    def _eligible_episode_names(self, episodes: List[EpisodeInfo], *, force: bool) -> List[str]:
        ids = set(self._eligible_episode_ids(episodes, force=force))
        return [info.name for info in episodes if info.id in ids]

    def _eligible_episode_ids(self, episodes: List[EpisodeInfo], *, force: bool) -> List[str]:
        protected = {} if force else self._protected_reasons(episodes)
        ids: List[str] = []
        with self._state_lock:
            states = dict(self._state.get("episodes", {}))
            current = self._current_episode
            queued = set(self._queued)
        name_counts = self._episode_name_counts(episodes)
        for info in episodes:
            if not info.complete:
                continue
            episode_id = info.id
            state = states.get(episode_id, self._legacy_state_for(info, name_counts))
            if state.get("status") == "success":
                continue
            if info.upload_record_status == "success":
                continue
            if self._failure_reason(info, state):
                continue
            if episode_id == current or episode_id in queued:
                continue
            if episode_id in protected:
                continue
            ids.append(episode_id)
        return ids

    def _protected_reasons(self, episodes: List[EpisodeInfo]) -> Dict[str, str]:
        complete = [item for item in episodes if item.complete]
        complete.sort(key=lambda item: item.mtime)
        newest = complete[-self.policy.keep_latest_count :] if self.policy.keep_latest_count > 0 else []
        newest_ids = {item.id for item in newest}
        cutoff = time.time() - self.policy.protect_minutes * 60.0
        reasons: Dict[str, str] = {}
        for item in complete:
            parts: List[str] = []
            if item.mtime > cutoff:
                parts.append(f"newer than {self.policy.protect_minutes} min")
            if item.id in newest_ids:
                parts.append(f"within latest {self.policy.keep_latest_count}")
            if parts:
                reasons[item.id] = ", ".join(parts)
        return reasons

    def _failed_file_tree(
        self,
        episodes: List[EpisodeInfo],
        persisted: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        name_counts = self._episode_name_counts(episodes)
        for info in episodes:
            reason = self._failure_reason(info, persisted.get(info.id, self._legacy_state_for(info, name_counts)))
            if not reason:
                continue
            rows.append(
                {
                    "id": info.id,
                    "episode_name": info.name,
                    "task_name": info.path.parent.name,
                    "task_path": str(info.path.parent),
                    "path": str(info.path),
                    "reason": reason,
                    "size_bytes": info.size_bytes,
                    "file_count": info.file_count,
                }
            )
        rows.sort(key=lambda item: (item["task_name"], item["episode_name"]))
        return rows

    def _failure_reason(self, info: EpisodeInfo, state: Dict[str, Any]) -> str:
        status = str(state.get("status") or "")
        if status == "success":
            return ""
        if status in {"queued", "uploading", "retrying"}:
            return ""
        if status == "failed":
            return str(state.get("last_error") or "upload failed")
        if status == "pending":
            return ""
        retry_count = int(state.get("retry_count") or 0)
        attempt_count = int(state.get("attempt_count") or 0)
        has_failure_history = attempt_count > 0 or bool(state.get("last_error"))
        if (
            has_failure_history
            and retry_count >= self.max_retries
            and status not in {"success", "queued", "uploading", "retrying"}
        ):
            return str(state.get("last_error") or f"retry count reached {retry_count}")
        if info.upload_record_status == "failed":
            return info.upload_record_note or "legacy upload.json failed"
        return ""

    def _upload_record_status(self, info: EpisodeInfo) -> str:
        upload_state = self._read_episode_upload_json(info.path)
        return str(upload_state.get("status") or "").lower()

    @staticmethod
    def _read_episode_upload_json(episode_path: Path) -> Dict[str, Any]:
        path = episode_path / "upload.json"
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"status": "failed", "note": "upload.json is unreadable"}
        return data if isinstance(data, dict) else {}

    def _enqueue_episode(
        self,
        episode_id: str,
        *,
        front: bool = False,
        max_attempts: Optional[int] = None,
    ) -> bool:
        return bool(
            self._enqueue_episode_ids(
                [episode_id],
                front=front,
                max_attempts=max_attempts,
            )
        )

    def _enqueue_episode_ids(
        self,
        episode_ids: Iterable[str],
        *,
        front: bool = False,
        max_attempts: Optional[int] = None,
        timings: Optional[Dict[str, float]] = None,
    ) -> int:
        """Queue multiple episodes and persist their state in one write."""
        count = 0
        lock_started = time.perf_counter()
        with self._queue_cond:
            if timings is not None:
                timings["queue_lock_wait_sec"] = time.perf_counter() - lock_started
            mutation_started = time.perf_counter()
            for episode_id in episode_ids:
                if episode_id == self._current_episode or episode_id in self._queued:
                    continue
                self._queued.add(episode_id)
                self._set_episode_state_locked(
                    episode_id,
                    save=False,
                    status="queued",
                    last_error="",
                )
                job = UploadJob(episode_id=episode_id, max_attempts=max_attempts)
                if front:
                    self._queue.appendleft(job)
                else:
                    self._queue.append(job)
                count += 1
            if count:
                self._save_state_locked()
                self._queue_cond.notify_all()
            if timings is not None:
                timings["queue_mutation_sec"] = time.perf_counter() - mutation_started
        return count

    def _set_episode_success(self, episode_id: str, result: UploadResult) -> None:
        files = [
            {
                "local_path": item.local_path,
                "object_key": item.object_key,
                "size_bytes": item.size_bytes,
                "skipped": item.skipped,
            }
            for item in result.files
        ]
        self._set_episode_state(
            episode_id,
            status="success",
            last_error="",
            oss={
                "bucket": result.bucket,
                "endpoint": result.endpoint,
                "prefix": result.prefix,
                "file_count": len(result.files),
                "files": files,
            },
        )

    def _state_for(self, episode_id: str) -> Dict[str, Any]:
        episodes = self._state.setdefault("episodes", {})
        return episodes.setdefault(episode_id, {})

    def _episode_path_for_id(self, episode_id: str) -> Path:
        with self._state_lock:
            cached = self._episode_cache.get(episode_id)
        if cached is not None:
            return cached.path
        for info in self.scan():
            if info.id == episode_id:
                return info.path
        return Path(episode_id)

    def _set_episode_state(self, episode_id: str, **updates: Any) -> None:
        with self._state_lock:
            self._set_episode_state_locked(episode_id, **updates)

    def _set_episode_state_locked(
        self,
        episode_id: str,
        *,
        save: bool = True,
        **updates: Any,
    ) -> None:
        state = self._state_for(episode_id)
        state.update(updates)
        state["updated_at"] = utc_now_iso()
        state.setdefault("attempt_count", 0)
        state.setdefault("retry_count", 0)
        if save:
            self._save_state_locked()

    @staticmethod
    def _episode_name_counts(episodes: List[EpisodeInfo]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for info in episodes:
            counts[info.name] = counts.get(info.name, 0) + 1
        return counts

    def _legacy_state_for(
        self,
        info: EpisodeInfo,
        name_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        if name_counts is not None and name_counts.get(info.name, 0) != 1:
            return {}
        legacy_state = self._state.get("episodes", {}).get(info.name, {})
        return legacy_state if isinstance(legacy_state, dict) else {}

    def _mark_interrupted_transient_states_locked(self) -> None:
        for state in self._state.get("episodes", {}).values():
            if state.get("status") in {"queued", "uploading", "retrying"}:
                state["status"] = "pending"
                state["last_error"] = "interrupted before completion"
                state["updated_at"] = utc_now_iso()

    def _replace_episode_cache(self, episodes: List[EpisodeInfo]) -> None:
        now = time.time()
        episode_ids = {episode.id for episode in episodes}
        episode_names = self._unique_episode_names(episodes)
        with self._state_lock:
            self._episode_cache = {episode.id: episode for episode in episodes}
            self._episode_cache_order = [episode.id for episode in episodes]
            self._episode_cache_refreshed_at = {episode.id: now for episode in episodes}
            self._episode_cache_initialized = True
            self._snapshot_refresh_index = 0
            self._last_snapshot_discovery = now
            self._remove_missing_episode_ids_locked(episode_ids, episode_names)

    def _reset_episode_cache_locked(self) -> None:
        self._episode_cache.clear()
        self._episode_cache_order.clear()
        self._episode_cache_refreshed_at.clear()
        self._episode_cache_initialized = False
        self._snapshot_refresh_index = 0
        self._last_snapshot_discovery = 0.0

    def _cache_episode(self, episode: EpisodeInfo, *, refreshed_at: Optional[float] = None) -> None:
        with self._state_lock:
            if episode.id not in self._episode_cache:
                self._episode_cache_order.append(episode.id)
            self._episode_cache[episode.id] = episode
            self._episode_cache_refreshed_at[episode.id] = time.time() if refreshed_at is None else refreshed_at

    def _refresh_cached_episode(self, episode_id: str) -> None:
        with self._scan_lock:
            with self._state_lock:
                cached = self._episode_cache.get(episode_id)
            path = cached.path if cached is not None else Path(episode_id)
            started_at = time.perf_counter()
            refreshed = scan_episode_dir(path)
            elapsed_sec = time.perf_counter() - started_at
            with self._state_lock:
                if refreshed is None:
                    self._episode_cache.pop(episode_id, None)
                    self._episode_cache_refreshed_at.pop(episode_id, None)
                    self._episode_cache_order = [item for item in self._episode_cache_order if item != episode_id]
                    return
                if refreshed.id != episode_id:
                    self._episode_cache.pop(episode_id, None)
                    self._episode_cache_refreshed_at.pop(episode_id, None)
                    self._episode_cache_order = [item for item in self._episode_cache_order if item != episode_id]
                    if refreshed.id not in self._episode_cache:
                        self._episode_cache_order.append(refreshed.id)
                self._episode_cache[refreshed.id] = refreshed
                self._episode_cache_refreshed_at[refreshed.id] = time.time()
        if elapsed_sec >= PERF_EPISODE_REFRESH_WARN_SEC:
            self._append_perf_event(
                "auto_upload_episode_refresh_slow",
                elapsed_sec=elapsed_sec,
                episode_path=str(path),
                file_count=refreshed.file_count if refreshed is not None else 0,
                size_bytes=refreshed.size_bytes if refreshed is not None else 0,
            )

    def _discover_snapshot_episode_dirs(self) -> None:
        with self._scan_lock:
            started_at = time.perf_counter()
            discovered: Dict[str, Path] = {}
            for data_dir in self.data_dirs:
                for episode_path in iter_episode_dirs(data_dir):
                    discovered[episode_key(episode_path)] = episode_path

            with self._state_lock:
                known = set(self._episode_cache.keys())
                stale = known - set(discovered.keys())
                name_counts = Counter(path.name for path in discovered.values())
                episode_names = {name for name, count in name_counts.items() if count == 1}
                for episode_id in stale:
                    self._episode_cache.pop(episode_id, None)
                    self._episode_cache_refreshed_at.pop(episode_id, None)
                self._episode_cache_order = [
                    episode_id for episode_id in self._episode_cache_order if episode_id in self._episode_cache
                ]
                self._remove_missing_episode_ids_locked(set(discovered.keys()), episode_names)
                new_paths = [path for episode_id, path in discovered.items() if episode_id not in known]

            for path in new_paths:
                episode = make_episode_placeholder(path)
                if episode is not None:
                    self._cache_episode(episode, refreshed_at=0.0)
        elapsed_sec = time.perf_counter() - started_at
        if elapsed_sec >= PERF_DISCOVERY_WARN_SEC:
            self._append_perf_event(
                "auto_upload_discovery_slow",
                elapsed_sec=elapsed_sec,
                discovered_count=len(discovered),
                new_count=len(new_paths),
                data_dirs=[str(path) for path in self.data_dirs],
            )

    def _next_snapshot_refresh_ids(self, current: Optional[str]) -> List[str]:
        with self._state_lock:
            order = [episode_id for episode_id in self._episode_cache_order if episode_id != current]
            if not order:
                return []
            start = self._snapshot_refresh_index % len(order)
            count = min(SNAPSHOT_INCREMENTAL_BATCH_SIZE, len(order))
            ids = [order[(start + offset) % len(order)] for offset in range(count)]
            self._snapshot_refresh_index = (start + count) % len(order)
            return ids

    def _state_path(self) -> Path:
        return self.data_dir / STATE_FILE_NAME

    def _append_perf_event(self, event: str, **fields: Any) -> None:
        now = time.monotonic()
        last_at = self._last_perf_event_at.get(event, 0.0)
        if now - last_at < PERF_EVENT_MIN_INTERVAL_SEC:
            return
        self._last_perf_event_at[event] = now
        payload = {
            "running": True,
            "source": "upload_manager",
            "pid": os.getpid(),
            "event": event,
            "updated_at": utc_now_iso(),
        }
        payload.update(fields)
        append_perf_log(payload, DEFAULT_PERF_LOG_PATH)

    def _clear_pending_queue_locked(self) -> None:
        self._queued.clear()
        self._queue.clear()

    @staticmethod
    def _unique_episode_names(episodes: List[EpisodeInfo]) -> Set[str]:
        counts = AutoUploadManager._episode_name_counts(episodes)
        return {name for name, count in counts.items() if count == 1}

    def _remove_missing_episode_ids_locked(self, live_episode_ids: Set[str], live_unique_names: Set[str]) -> None:
        live_refs = live_episode_ids | live_unique_names
        missing_queued = self._queued - live_refs
        states = self._state.get("episodes", {})
        missing_states: Set[str] = set()
        if isinstance(states, dict):
            missing_states = set(states) - live_refs
            if self._current_episode:
                missing_states.discard(self._current_episode)
        if not missing_queued and not missing_states:
            return
        self._queued.difference_update(missing_queued)
        self._queue = deque(job for job in self._queue if job.episode_id not in missing_queued)
        if isinstance(states, dict):
            for episode_id in missing_states | missing_queued:
                states.pop(episode_id, None)
        self._save_state_locked()

    def _forget_missing_episode(self, episode_id: str) -> None:
        with self._state_lock:
            self._queued.discard(episode_id)
            self._queue = deque(job for job in self._queue if job.episode_id != episode_id)
            self._episode_cache.pop(episode_id, None)
            self._episode_cache_refreshed_at.pop(episode_id, None)
            self._episode_cache_order = [item for item in self._episode_cache_order if item != episode_id]
            states = self._state.get("episodes", {})
            if isinstance(states, dict):
                states.pop(episode_id, None)
            self._save_state_locked()

    def episode_cache_initialized(self) -> bool:
        with self._state_lock:
            return self._episode_cache_initialized

    def _cached_episodes(self) -> List[EpisodeInfo]:
        with self._state_lock:
            episodes = list(self._episode_cache.values())
        episodes.sort(key=lambda item: item.mtime)
        return episodes

    def _prune_missing_protection_entries(self, episodes: List[EpisodeInfo]) -> List[EpisodeInfo]:
        keep_count = self.policy.keep_latest_count
        if keep_count <= 0:
            return episodes

        live_count = 0
        missing_ids: Set[str] = set()
        for episode in sorted(
            (item for item in episodes if item.complete),
            key=lambda item: item.mtime,
            reverse=True,
        ):
            if live_count >= keep_count:
                break
            if is_complete_episode_dir(episode.path):
                live_count += 1
            else:
                missing_ids.add(episode.id)

        for episode_id in missing_ids:
            self._forget_missing_episode(episode_id)
        if not missing_ids:
            return episodes
        return [episode for episode in episodes if episode.id not in missing_ids]

    def _validate_managed_episode_path(self, episode_path: Path) -> Path:
        target = episode_path.expanduser().resolve()
        if not target.name.startswith("episode_") or target.name.endswith("_temp"):
            raise ValueError(f"Invalid finalized episode path: {target}")
        with self._state_lock:
            data_dirs = list(self.data_dirs)
        if not any(target.is_relative_to(data_dir) for data_dir in data_dirs):
            raise ValueError(f"Episode path is outside configured data directories: {target}")
        return target

    def _load_state(self) -> None:
        with self._state_lock:
            path = self._state_path()
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        self._state = data
                    else:
                        self._state = {"version": 1, "episodes": {}}
                except Exception as exc:
                    logger.warning(f"Failed to load upload state {path}: {exc}")
                    self._state = {"version": 1, "episodes": {}}
            else:
                self._state = {"version": 1, "episodes": {}}
            if "resource_type" in self._state:
                self.resource_type = str(self._state.get("resource_type") or "")
            policy = self._state.get("policy") if isinstance(self._state.get("policy"), dict) else {}
            self.policy = UploadPolicy(
                protect_minutes=int(policy.get("protect_minutes", self.policy.protect_minutes)),
                keep_latest_count=int(policy.get("keep_latest_count", self.policy.keep_latest_count)),
            )
            self.poll_interval_sec = 0
            self.max_retries = int(policy.get("max_retries", self.max_retries))
            self.upload_workers = int(policy.get("upload_workers", self.upload_workers))
            self.multipart_threads = int(policy.get("multipart_threads", self.multipart_threads))
            self.upload_backend = str(policy.get("upload_backend") or self.upload_backend)
            if self.upload_backend not in {UPLOAD_BACKEND_OSS, UPLOAD_BACKEND_HTTP_SINK}:
                self.upload_backend = UPLOAD_BACKEND_OSS
            self.http_sink_url = str(policy.get("http_sink_url") or self.http_sink_url)
            self.http_timeout_sec = int(policy.get("http_timeout_sec", self.http_timeout_sec))
            saved_dirs = self._state.get("data_dirs")
            if isinstance(saved_dirs, list):
                self._set_scan_dirs([*self.data_dirs, *(Path(str(item)) for item in saved_dirs)])
            self._state.setdefault("version", 1)
            self._state.setdefault("episodes", {})

    def _save_state_locked(self) -> int:
        """Mark the current state dirty without performing filesystem I/O."""
        self._state["data_dir"] = str(self.data_dir)
        self._state["data_dirs"] = [str(path) for path in self.data_dirs]
        self._state_revision += 1
        revision = self._state_revision
        self._persistence_event.set()
        return revision

    @staticmethod
    def _persist_state_snapshot(path: Path, state: Dict[str, Any]) -> Dict[str, float]:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        serialize_started = time.perf_counter()
        serialized = json.dumps(
            state,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        serialize_sec = time.perf_counter() - serialize_started
        write_started = time.perf_counter()
        tmp.write_text(serialized + "\n", encoding="utf-8")
        write_sec = time.perf_counter() - write_started
        replace_started = time.perf_counter()
        tmp.replace(path)
        replace_sec = time.perf_counter() - replace_started
        return {
            "serialize_sec": serialize_sec,
            "write_sec": write_sec,
            "replace_sec": replace_sec,
            "bytes": float(len(serialized.encode("utf-8")) + 1),
        }

    @staticmethod
    def _persist_status_snapshot(paths: List[Path], payload: Dict[str, Any]) -> None:
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(serialized, encoding="utf-8")
            tmp.replace(path)

    def _persistence_loop(self) -> None:
        while True:
            self._persistence_event.wait()
            self._persistence_event.clear()

            while True:
                with self._state_lock:
                    state_revision = self._state_revision
                    state_pending = state_revision > self._persisted_state_revision
                    state_path = self._state_path()
                    state_snapshot = copy.deepcopy(self._state) if state_pending else None

                    status_revision = self._status_revision
                    status_pending = status_revision > self._persisted_status_revision
                    status_payload = (
                        copy.deepcopy(self._pending_status_payload)
                        if status_pending and self._pending_status_payload is not None
                        else None
                    )
                    status_paths = list(self._pending_status_paths) if status_pending else []

                if not state_pending and not status_pending:
                    break

                try:
                    state_metrics: Dict[str, float] = {}
                    state_persist_started = 0.0
                    if state_snapshot is not None:
                        state_persist_started = time.perf_counter()
                        state_metrics = self._persist_state_snapshot(
                            state_path,
                            state_snapshot,
                        ) or {}
                    if status_payload is not None:
                        self._persist_status_snapshot(status_paths, status_payload)
                except Exception as exc:
                    logger.warning(f"Failed to persist upload state snapshot: {exc}")
                    if self._persistence_stop_event.wait(1.0):
                        return
                    self._persistence_event.set()
                    break

                with self._persistence_cond:
                    if state_snapshot is not None:
                        self._persisted_state_revision = max(
                            self._persisted_state_revision,
                            state_revision,
                        )
                    if status_payload is not None:
                        self._persisted_status_revision = max(
                            self._persisted_status_revision,
                            status_revision,
                        )
                    self._persistence_cond.notify_all()

                state_elapsed = (
                    time.perf_counter() - state_persist_started
                    if state_persist_started
                    else 0.0
                )
                if state_elapsed >= PERF_EVENT_WARN_SEC:
                    self._append_perf_event(
                        "upload_state_persist_slow",
                        revision=state_revision,
                        elapsed_sec=state_elapsed,
                        **state_metrics,
                    )

            if self._persistence_stop_event.is_set():
                with self._state_lock:
                    if (
                        self._persisted_state_revision >= self._state_revision
                        and self._persisted_status_revision >= self._status_revision
                    ):
                        return

    def flush_persistence(self, *, timeout: float = 5.0) -> bool:
        """Wait until state/status revisions visible at call time reach disk."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._persistence_cond:
            state_revision = self._state_revision
            status_revision = self._status_revision
            self._persistence_event.set()
            while (
                self._persisted_state_revision < state_revision
                or self._persisted_status_revision < status_revision
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._persistence_cond.wait(timeout=remaining)
        return True

    def _upload_status_path(self) -> Path:
        return self.data_dir / UPLOAD_STATUS_FILE_NAME

    def _upload_status_paths(self) -> List[Path]:
        paths = [self._upload_status_path()]
        if DEFAULT_UPLOAD_STATUS_PATH is not None:
            default_path = DEFAULT_UPLOAD_STATUS_PATH.expanduser().resolve()
            if default_path not in {path.expanduser().resolve() for path in paths}:
                paths.append(default_path)
        return paths

    def _write_status_snapshot_locked(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_status_write_at < UPLOAD_STATUS_WRITE_INTERVAL_SEC:
            return
        self._last_status_write_at = now
        progress = self._progress_dict(self._current_progress)
        current_episode = self._current_episode
        episode_path = ""
        episode_name = ""
        if current_episode:
            cached = self._episode_cache.get(current_episode)
            path = cached.path if cached is not None else Path(current_episode)
            episode_path = str(path)
            episode_name = path.name
        total_bytes = int((progress or {}).get("total_bytes") or 0)
        current_bytes = int((progress or {}).get("current_bytes") or (progress or {}).get("completed_bytes") or 0)
        percent = (current_bytes / total_bytes * 100.0) if total_bytes > 0 else 0.0
        payload = {
            "updated_at": utc_now_iso(),
            "data_dir": str(self.data_dir),
            "auto_enabled": self._auto_event.is_set(),
            "worker_running": self._worker_thread is not None and self._worker_thread.is_alive(),
            "queue_count": len(self._queue),
            "queued_count": len(self._queued),
            "current_episode": current_episode,
            "episode_name": episode_name,
            "episode_path": episode_path,
            "progress": progress,
            "percent": round(percent, 2),
        }
        self._pending_status_payload = payload
        self._pending_status_paths = self._upload_status_paths()
        self._status_revision += 1
        self._persistence_event.set()

    def _progress_dict(self, progress: Optional[UploadProgress]) -> Optional[Dict[str, int]]:
        if progress is None:
            return None
        return {
            "completed_files": progress.completed_files,
            "total_files": progress.total_files,
            "completed_bytes": progress.completed_bytes,
            "total_bytes": progress.total_bytes,
            "current_bytes": progress.current_bytes,
        }

    def _episode_progress(
        self,
        episode_id: str,
        status: str,
        current_progress: Optional[Dict[str, int]],
    ) -> Dict[str, int]:
        with self._state_lock:
            state = dict(self._state.get("episodes", {}).get(episode_id, {}))
        if status == "uploading" and current_progress is not None:
            return current_progress
        return {
            "completed_files": int(state.get("completed_files") or 0),
            "total_files": int(state.get("total_files") or 0),
            "completed_bytes": int(state.get("completed_bytes") or 0),
            "total_bytes": int(state.get("total_bytes") or 0),
            "current_bytes": int(state.get("current_bytes") or 0),
        }

    def _log(self, message: str) -> None:
        if self.log_callback is not None:
            self.log_callback(f"{datetime.now().strftime('%H:%M:%S')} {message}\n")

    def _set_scan_dirs(self, paths: Iterable[Path]) -> None:
        seen: Set[Path] = set()
        data_dirs: List[Path] = []
        for path in paths:
            resolved = Path(path).expanduser().resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            data_dirs.append(resolved)
        if not data_dirs:
            data_dirs = [self.data_dir]
        self.data_dirs = data_dirs


def scan_episode_dir(episode_dir: Path) -> Optional[EpisodeInfo]:
    child = episode_dir.expanduser().resolve()
    if not child.is_dir():
        return None
    try:
        files = [path for path in child.rglob("*") if path.is_file()]
        size_bytes = sum(path.stat().st_size for path in files)
        complete = is_complete_episode_dir(child)
        mtime = child.stat().st_mtime
    except FileNotFoundError:
        return None
    if not child.is_dir():
        return None
    upload_record = read_episode_upload_record(child)
    return EpisodeInfo(
        name=child.name,
        path=child,
        mtime=mtime,
        size_bytes=size_bytes,
        file_count=len(files),
        complete=complete,
        upload_record_status=str(upload_record.get("status") or "").lower(),
        upload_record_note=str(upload_record.get("note") or upload_record.get("last_error") or ""),
    )


def is_complete_episode_dir(episode_dir: Path) -> bool:
    child = episode_dir.expanduser().resolve()
    try:
        if not child.is_dir() or not (child / "metadata.json").is_file():
            return False
        if (child / "data.mcap").is_file():
            return True
        chunk_dir = child / "data"
        return (chunk_dir / "mcap_index.json").is_file() and any(chunk_dir.glob("*.mcap"))
    except FileNotFoundError:
        return False


def make_episode_placeholder(episode_dir: Path) -> Optional[EpisodeInfo]:
    child = episode_dir.expanduser().resolve()
    if not child.is_dir():
        return None
    try:
        mtime = child.stat().st_mtime
    except OSError:
        mtime = 0.0
    upload_record = read_episode_upload_record(child)
    return EpisodeInfo(
        name=child.name,
        path=child,
        mtime=mtime,
        size_bytes=0,
        file_count=0,
        complete=False,
        upload_record_status=str(upload_record.get("status") or "").lower(),
        upload_record_note=str(upload_record.get("note") or upload_record.get("last_error") or ""),
        details_loaded=False,
    )


def scan_episode_dirs(data_dir: Path) -> List[EpisodeInfo]:
    episodes: List[EpisodeInfo] = []
    for child in iter_episode_dirs(data_dir):
        episode = scan_episode_dir(child)
        if episode is not None:
            episodes.append(episode)
    episodes.sort(key=lambda item: item.mtime)
    return episodes


def iter_episode_dirs(data_dir: Path) -> Iterable[Path]:
    root = data_dir.expanduser().resolve()
    if not root.is_dir():
        return
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("episode_") or child.name.endswith("_temp"):
            for grandchild in child.iterdir():
                if (
                    grandchild.is_dir()
                    and grandchild.name.startswith("episode_")
                    and not grandchild.name.endswith("_temp")
                ):
                    yield grandchild
            continue
        yield child


def read_episode_upload_record(episode_path: Path) -> Dict[str, Any]:
    path = episode_path / "upload.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "failed", "note": "upload.json is unreadable"}
    return data if isinstance(data, dict) else {}


def scan_episode_dirs_many(data_dirs: Iterable[Path]) -> List[EpisodeInfo]:
    episodes: List[EpisodeInfo] = []
    seen_paths: Set[Path] = set()
    for data_dir in data_dirs:
        for episode in scan_episode_dirs(data_dir):
            resolved = episode.path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            episodes.append(episode)
    episodes.sort(key=lambda item: item.mtime)
    return episodes
