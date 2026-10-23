#!/usr/bin/env python3
"""
Aliyun OSS uploader for DM3 Windows local collection data.

Author: yichen Xie
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.logger_config import logger
from incubator.nedf3.windows_DM3.oss_upload_config import (
    DEFAULT_OSS_MULTIPART_UPLOAD_CONFIG,
    OssMultipartUploadConfig,
)


ProgressCallback = Callable[[str], None]
StsTokenRefreshCallback = Callable[[], Dict[str, Any]]

INITIAL_RETRY_DELAY_SEC = 1.0
MAX_RETRY_DELAY_SEC = 60.0
DEFAULT_MAX_UPLOAD_WORKERS = 4
_OSS_MIN_PART_SIZE_BYTES = 100 * 1024
_OSS_MAX_PART_COUNT = 10000


class UploadCancelled(RuntimeError):
    """Raised when an in-flight upload is cancelled by the caller."""


@dataclass
class UploadProgress:
    """Aggregate upload progress across parallel workers."""

    completed_files: int
    total_files: int
    completed_bytes: int
    total_bytes: int
    current_bytes: int


UploadProgressCallback = Callable[[UploadProgress], None]


@dataclass
class UploadedFile:
    """One uploaded local file and its OSS object key."""

    local_path: str
    object_key: str
    size_bytes: int
    skipped: bool = False


@dataclass
class UploadResult:
    """Summary of one upload run."""

    bucket: str
    endpoint: str
    prefix: str
    files: List[UploadedFile]


@dataclass(frozen=True)
class _UploadTask:
    """One local file scheduled for OSS upload."""

    file_path: Path
    object_key: str
    local_size: int


def upload_episode_data_dir(
    data_dir: Path,
    sts_token: Dict[str, Any],
    *,
    progress: Optional[ProgressCallback] = None,
    refresh_sts_token: Optional[StsTokenRefreshCallback] = None,
    max_workers: int = DEFAULT_MAX_UPLOAD_WORKERS,
    on_upload_progress: Optional[UploadProgressCallback] = None,
    multipart_config: Optional[OssMultipartUploadConfig] = None,
    stop_event: Optional[threading.Event] = None,
) -> UploadResult:
    """
    Upload a local task directory or a single episode directory to OSS.

    Object layout:
        {sts.dir}/{episode_uuid}/{relative_file_path}

    Local episode directories keep the ``episode_`` prefix for discovery, but
    the DM3 OSS object prefix omits it.

    If ``data_dir`` itself is an episode directory, only that episode is uploaded.
    Otherwise direct child directories named ``episode_*`` are uploaded. Temporary
    directories ending in ``_temp`` are skipped.

    When ``refresh_sts_token`` is provided, expired STS credentials are refreshed
    automatically and the current file upload is retried.

    Files already present on OSS with a matching size are skipped so retries can
    resume after partial failures without re-uploading completed objects.

    Uploads run in parallel when ``max_workers`` is greater than 1.

    Files at or above ``multipart_config.multipart_threshold_bytes`` use
    ``oss2.resumable_upload`` (multipart + per-file ``num_threads``).
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")

    resolved_multipart_config = (
        multipart_config or DEFAULT_OSS_MULTIPART_UPLOAD_CONFIG
    )

    oss2 = _import_oss2()
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist or is not a directory: {data_dir}")

    normalized_token = _normalize_sts_token(sts_token)
    bucket_name, endpoint, prefix = _parse_sts_location(normalized_token)
    session = _OssUploadSession(
        oss2=oss2,
        sts_token=normalized_token,
        refresh_sts_token=refresh_sts_token,
        multipart_config=resolved_multipart_config,
        stop_event=stop_event,
    )

    episode_dirs = _discover_episode_dirs(data_dir)
    if not episode_dirs:
        raise ValueError(f"No finalized episode directories found under: {data_dir}")

    upload_tasks = _collect_upload_tasks(episode_dirs, prefix)
    if not upload_tasks:
        raise ValueError(f"No files found to upload under: {data_dir}")

    progress_tracker = _UploadProgressTracker(upload_tasks, on_upload_progress)
    progress_tracker.emit()

    worker_count = min(max_workers, len(upload_tasks))
    if worker_count > 1:
        _emit(progress, f"Uploading {len(upload_tasks)} files with {worker_count} parallel workers...")
        uploaded_files = _upload_tasks_parallel(
            session,
            upload_tasks,
            bucket_name=bucket_name,
            progress=progress,
            progress_tracker=progress_tracker,
            max_workers=worker_count,
            stop_event=stop_event,
        )
    else:
        uploaded_files = _upload_tasks_sequential(
            session,
            upload_tasks,
            bucket_name=bucket_name,
            progress=progress,
            progress_tracker=progress_tracker,
            stop_event=stop_event,
        )

    skipped_count = sum(1 for item in uploaded_files if item.skipped)
    uploaded_count = len(uploaded_files) - skipped_count
    logger.info(
        f"Uploaded {uploaded_count} files, skipped {skipped_count} existing files from "
        f"{data_dir} to oss://{bucket_name}/{prefix} "
        f"(local episode_<uuid> -> OSS <uuid>, workers={worker_count})"
    )
    return UploadResult(
        bucket=bucket_name,
        endpoint=endpoint,
        prefix=prefix,
        files=uploaded_files,
    )


def _collect_upload_tasks(episode_dirs: List[Path], prefix: str) -> List[_UploadTask]:
    tasks: List[_UploadTask] = []
    for episode_dir in episode_dirs:
        upload_episode_name = _upload_episode_name(episode_dir.name)
        for file_path in _iter_upload_files(episode_dir):
            relative_path = file_path.relative_to(episode_dir)
            object_key = _make_object_key(prefix, upload_episode_name, relative_path)
            tasks.append(
                _UploadTask(
                    file_path=file_path,
                    object_key=object_key,
                    local_size=file_path.stat().st_size,
                )
            )
    return tasks


class _UploadProgressTracker:
    """Thread-safe aggregate upload progress for parallel transfers."""

    def __init__(
        self,
        upload_tasks: List[_UploadTask],
        callback: Optional[UploadProgressCallback],
    ) -> None:
        self._callback = callback
        self._lock = threading.Lock()
        self.total_files = len(upload_tasks)
        self.total_bytes = sum(task.local_size for task in upload_tasks)
        self.completed_files = 0
        self.completed_bytes = 0
        self._inflight_bytes: Dict[str, int] = {}
        self._last_emit_at = 0.0

    def start_file(self, object_key: str) -> None:
        with self._lock:
            self._inflight_bytes[object_key] = 0
        self.emit(force=True)

    def update_file(self, object_key: str, consumed_bytes: int) -> None:
        with self._lock:
            self._inflight_bytes[object_key] = consumed_bytes
        self.emit()

    def finish_file(self, object_key: str, size_bytes: int) -> None:
        with self._lock:
            self._inflight_bytes.pop(object_key, None)
            self.completed_files += 1
            self.completed_bytes += size_bytes
        self.emit(force=True)

    def cancel_file(self, object_key: str) -> None:
        with self._lock:
            self._inflight_bytes.pop(object_key, None)
        self.emit(force=True)

    def snapshot(self) -> UploadProgress:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> UploadProgress:
        current_bytes = self.completed_bytes + sum(self._inflight_bytes.values())
        return UploadProgress(
            completed_files=self.completed_files,
            total_files=self.total_files,
            completed_bytes=self.completed_bytes,
            total_bytes=self.total_bytes,
            current_bytes=current_bytes,
        )

    def emit(self, *, force: bool = False) -> None:
        if self._callback is None:
            return
        now = time.monotonic()
        with self._lock:
            if not force and now - self._last_emit_at < 0.25:
                return
            self._last_emit_at = now
            progress = self._snapshot_locked()
        self._callback(progress)


def _upload_tasks_sequential(
    session: "_OssUploadSession",
    upload_tasks: List[_UploadTask],
    *,
    bucket_name: str,
    progress: Optional[ProgressCallback],
    progress_tracker: _UploadProgressTracker,
    stop_event: Optional[threading.Event],
) -> List[UploadedFile]:
    return [
        _upload_one_file(
            session,
            task,
            bucket_name=bucket_name,
            progress=progress,
            progress_tracker=progress_tracker,
            stop_event=stop_event,
        )
        for task in upload_tasks
    ]


def _upload_tasks_parallel(
    session: "_OssUploadSession",
    upload_tasks: List[_UploadTask],
    *,
    bucket_name: str,
    progress: Optional[ProgressCallback],
    progress_tracker: _UploadProgressTracker,
    max_workers: int,
    stop_event: Optional[threading.Event],
) -> List[UploadedFile]:
    results: List[Optional[UploadedFile]] = [None] * len(upload_tasks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                _upload_one_file,
                session,
                task,
                bucket_name=bucket_name,
                progress=progress,
                progress_tracker=progress_tracker,
                stop_event=stop_event,
            ): index
            for index, task in enumerate(upload_tasks)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
    return [item for item in results if item is not None]


def _upload_one_file(
    session: "_OssUploadSession",
    task: _UploadTask,
    *,
    bucket_name: str,
    progress: Optional[ProgressCallback],
    progress_tracker: _UploadProgressTracker,
    stop_event: Optional[threading.Event],
) -> UploadedFile:
    _raise_if_cancelled(stop_event)
    progress_tracker.start_file(task.object_key)
    try:
        _raise_if_cancelled(stop_event)
        if session.is_already_uploaded(task.object_key, task.local_size, progress=progress):
            _emit(
                progress,
                f"Skipping {task.file_path} (already on oss://{bucket_name}/{task.object_key})",
            )
            result = UploadedFile(
                local_path=str(task.file_path),
                object_key=task.object_key,
                size_bytes=task.local_size,
                skipped=True,
            )
            progress_tracker.finish_file(task.object_key, task.local_size)
            return result

        upload_mode = (
            "multipart"
            if task.local_size >= session.multipart_config.multipart_threshold_bytes
            else "single"
        )
        _emit(
            progress,
            f"Uploading ({upload_mode}) {task.file_path} -> "
            f"oss://{bucket_name}/{task.object_key}",
        )

        def file_byte_progress(consumed_bytes: int, _total_bytes: int) -> None:
            _raise_if_cancelled(stop_event)
            progress_tracker.update_file(task.object_key, consumed_bytes)

        _raise_if_cancelled(stop_event)
        session.put_object_from_file(
            task.object_key,
            task.file_path,
            progress=progress,
            byte_progress=file_byte_progress,
        )
        result = UploadedFile(
            local_path=str(task.file_path),
            object_key=task.object_key,
            size_bytes=task.local_size,
        )
        progress_tracker.finish_file(task.object_key, task.local_size)
        return result
    except Exception:
        progress_tracker.cancel_file(task.object_key)
        raise


def _import_oss2() -> Any:
    try:
        import oss2
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'oss2'. Install it with: pip install oss2"
        ) from exc
    return oss2


@dataclass
class _OssUploadSession:
    """OSS upload session that can refresh expired STS credentials."""

    oss2: Any
    sts_token: Dict[str, Any]
    refresh_sts_token: Optional[StsTokenRefreshCallback]
    multipart_config: OssMultipartUploadConfig = DEFAULT_OSS_MULTIPART_UPLOAD_CONFIG
    stop_event: Optional[threading.Event] = None
    bucket: Any = None
    _credentials_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _request_slots: threading.BoundedSemaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.bucket = _create_bucket(self.oss2, self.sts_token)
        self._request_slots = threading.BoundedSemaphore(
            self.multipart_config.max_inflight_requests
        )

    @contextmanager
    def _request_slot(self):
        self._request_slots.acquire()
        try:
            yield
        finally:
            self._request_slots.release()

    def _current_bucket(self) -> Any:
        with self._credentials_lock:
            return self.bucket

    def is_already_uploaded(
        self,
        object_key: str,
        local_size_bytes: int,
        *,
        progress: Optional[ProgressCallback] = None,
    ) -> bool:
        """Return True when OSS already has the object with the same byte size."""
        transient_attempt = 0
        while True:
            try:
                with self._request_slot():
                    meta = self._current_bucket().head_object(object_key)
                remote_size = int(meta.content_length or 0)
                return remote_size == local_size_bytes
            except self.oss2.exceptions.NoSuchKey:
                return False
            except self.oss2.exceptions.ServerError as exc:
                if exc.status == 404:
                    return False
                if _is_security_token_expired(exc):
                    self._refresh_credentials_with_progress(
                        progress,
                        context=f"checking {object_key}",
                    )
                    continue
                raise
            except Exception as exc:
                if not _is_transient_upload_error(exc):
                    raise
                transient_attempt += 1
                _sleep_for_transient_retry(
                    exc,
                    transient_attempt,
                    progress=progress,
                    context=f"skip check for {object_key}",
                )

    def put_object_from_file(
        self,
        object_key: str,
        file_path: Path,
        *,
        progress: Optional[ProgressCallback] = None,
        byte_progress: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Upload one local file, refreshing STS credentials on token expiry."""
        local_size = file_path.stat().st_size
        use_resumable = (
            local_size >= self.multipart_config.multipart_threshold_bytes
        )
        transient_attempt = 0
        while True:
            _raise_if_cancelled(self.stop_event)
            try:
                if use_resumable:
                    self._resumable_upload_file(
                        object_key,
                        file_path,
                        byte_progress=byte_progress,
                    )
                else:
                    with file_path.open("rb") as file_obj:
                        with self._request_slot():
                            self._current_bucket().put_object(
                                object_key,
                                file_obj,
                                progress_callback=byte_progress,
                            )
                return
            except self.oss2.exceptions.ServerError as exc:
                if not _is_security_token_expired(exc):
                    raise
                self._refresh_credentials_with_progress(
                    progress,
                    context=f"uploading {file_path}",
                )
            except Exception as exc:
                if not _is_transient_upload_error(exc):
                    raise
                transient_attempt += 1
                _raise_if_cancelled(self.stop_event)
                _sleep_for_transient_retry(
                    exc,
                    transient_attempt,
                    progress=progress,
                    context=f"uploading {file_path.name}",
                )

    def _resumable_upload_file(
        self,
        object_key: str,
        file_path: Path,
        *,
        byte_progress: Optional[Callable[[int, int], None]],
    ) -> None:
        """Multipart upload for large files with byte-level progress when requested."""
        if byte_progress is not None:
            self._multipart_upload_with_progress(
                object_key,
                file_path,
                byte_progress=byte_progress,
            )
            return

        cfg = self.multipart_config
        self.oss2.resumable_upload(
            self._current_bucket(),
            object_key,
            str(file_path),
            multipart_threshold=cfg.multipart_threshold_bytes,
            part_size=cfg.part_size,
            num_threads=cfg.num_threads,
            progress_callback=byte_progress,
        )

    def _multipart_upload_with_progress(
        self,
        object_key: str,
        file_path: Path,
        *,
        byte_progress: Callable[[int, int], None],
    ) -> None:
        """Upload file parts concurrently while reporting aggregate progress."""
        bucket = self._current_bucket()
        cfg = self.multipart_config
        file_size = file_path.stat().st_size
        part_size = _determine_part_size(self.oss2, file_size, cfg.part_size)
        with self._request_slot():
            upload_id = bucket.init_multipart_upload(object_key).upload_id

        part_specs = []
        offset = 0
        while offset < file_size:
            size = min(part_size, file_size - offset)
            part_specs.append((len(part_specs) + 1, offset, size))
            offset += size

        progress_lock = threading.Lock()
        part_progress: Dict[int, int] = {part_number: 0 for part_number, _, _ in part_specs}

        def upload_part(part_number: int, part_offset: int, size: int) -> Any:
            _raise_if_cancelled(self.stop_event)
            with self._request_slot():
                _raise_if_cancelled(self.stop_event)
                with file_path.open("rb") as handle:
                    handle.seek(part_offset)
                    chunk = handle.read(size)
                if len(chunk) != size:
                    raise OSError(
                        f"Short read for {file_path}: expected {size}, got {len(chunk)}"
                    )

                def report_part_progress(consumed: int, _total: int) -> None:
                    with progress_lock:
                        part_progress[part_number] = max(
                            part_progress[part_number], min(consumed, size)
                        )
                        total_consumed = sum(part_progress.values())
                        byte_progress(total_consumed, file_size)

                payload = self.oss2.utils.make_progress_adapter(
                    chunk,
                    report_part_progress,
                    size=size,
                )
                result = bucket.upload_part(
                    object_key, upload_id, part_number, payload
                )
                report_part_progress(size, size)
                return self.oss2.models.PartInfo(
                    part_number,
                    result.etag,
                    size=size,
                )

        try:
            worker_count = min(cfg.num_threads, len(part_specs))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(upload_part, *spec) for spec in part_specs]
                parts = [future.result() for future in as_completed(futures)]
            parts.sort(key=lambda part: part.part_number)
            with self._request_slot():
                bucket.complete_multipart_upload(object_key, upload_id, parts)
        except Exception:
            try:
                with self._request_slot():
                    bucket.abort_multipart_upload(object_key, upload_id)
            except Exception:
                pass
            raise

    def _refresh_credentials(self) -> None:
        _raise_if_cancelled(self.stop_event)
        if self.refresh_sts_token is None:
            raise RuntimeError("STS token expired and no refresh callback was provided.")
        with self._credentials_lock:
            self.sts_token = _normalize_sts_token(self.refresh_sts_token())
            self.bucket = _create_bucket(self.oss2, self.sts_token)

    def _refresh_credentials_with_progress(
        self,
        progress: Optional[ProgressCallback],
        *,
        context: str,
    ) -> None:
        _raise_if_cancelled(self.stop_event)
        _emit(
            progress,
            f"STS token expired while {context}; refreshing credentials and retrying...",
        )
        logger.warning(f"OSS STS token expired while {context}; refreshing credentials")
        self._refresh_credentials()


def _raise_if_cancelled(stop_event: Optional[threading.Event]) -> None:
    if stop_event is not None and stop_event.is_set():
        raise UploadCancelled("upload cancelled")


def _determine_part_size(oss2: Any, total_size: int, preferred_size: int) -> int:
    """Resolve multipart part size across oss2 versions."""
    if total_size <= 0:
        return preferred_size
    if total_size < preferred_size:
        return total_size

    resumable = getattr(oss2, "resumable", None)
    candidates = (
        getattr(oss2, "determine_part_size", None),
        getattr(resumable, "determine_part_size", None) if resumable is not None else None,
        getattr(getattr(oss2, "utils", None), "determine_part_size", None),
    )
    for determine in candidates:
        if determine is None:
            continue
        try:
            return int(determine(total_size, preferred_size))
        except TypeError:
            return int(determine(total_size, excepted_part_size=preferred_size))

    part_size = preferred_size
    while part_size * _OSS_MAX_PART_COUNT < total_size or part_size < _OSS_MIN_PART_SIZE_BYTES:
        part_size *= 2
    return part_size


def _normalize_sts_token(sts_token: Dict[str, Any]) -> Dict[str, Any]:
    """Extract STS credentials from common DM3 response shapes."""
    data = sts_token.get("data")
    if isinstance(data, dict) and data.get("access_key_id"):
        return data
    return sts_token


def _parse_sts_location(sts_token: Dict[str, Any]) -> Tuple[str, str, str]:
    bucket_name = _required_str(sts_token, "bucket")
    endpoint = _required_str(sts_token, "endpoint")
    prefix = _normalize_prefix(_required_str(sts_token, "dir"))
    return bucket_name, endpoint, prefix


def _create_bucket(oss2: Any, sts_token: Dict[str, Any]) -> Any:
    access_key_id = _required_str(sts_token, "access_key_id")
    access_key_secret = _required_str(sts_token, "access_key_secret")
    security_token = _required_str(sts_token, "security_token")
    bucket_name = _required_str(sts_token, "bucket")
    endpoint = _required_str(sts_token, "endpoint")
    auth = oss2.StsAuth(access_key_id, access_key_secret, security_token)
    return oss2.Bucket(auth, _normalize_endpoint(endpoint), bucket_name)


def _is_security_token_expired(exc: Exception) -> bool:
    status = getattr(exc, "status", None)
    details = getattr(exc, "details", None)
    if status != 403 or not isinstance(details, dict):
        return False
    return str(details.get("Code", "")) == "SecurityTokenExpired"


_TRANSIENT_ERROR_MARKERS: Tuple[str, ...] = (
    "SSLError",
    "SSLEOFError",
    "ConnectionError",
    "Connection reset",
    "Read timed out",
    "Max retries exceeded",
    "EOF occurred",
    "Connection aborted",
    "RemoteDisconnected",
    "Broken pipe",
)


def _is_transient_upload_error(exc: Exception) -> bool:
    """Return True for retryable network / TLS failures during OSS upload."""
    status = getattr(exc, "status", None)
    if status == -2:
        return True
    message = str(getattr(exc, "details", exc))
    return any(marker in message for marker in _TRANSIENT_ERROR_MARKERS)


def _retry_delay(attempt: int) -> float:
    return min(INITIAL_RETRY_DELAY_SEC * (2 ** attempt), MAX_RETRY_DELAY_SEC)


def _sleep_for_transient_retry(
    exc: Exception,
    attempt: int,
    *,
    progress: Optional[ProgressCallback],
    context: str,
) -> None:
    delay = _retry_delay(attempt - 1)
    _emit(
        progress,
        f"Network error while {context}; retry #{attempt} in {delay:.0f}s...",
    )
    logger.warning(
        f"Transient OSS error while {context}: {exc}; retry #{attempt} in {delay:.0f}s"
    )
    time.sleep(delay)


def _required_str(data: Dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not value:
        raise ValueError(f"STS token response missing required field: {key}")
    return str(value)


def _normalize_endpoint(endpoint: str) -> str:
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    return f"https://{endpoint}"


def _normalize_prefix(prefix: str) -> str:
    return prefix.strip().strip("/")


def _discover_episode_dirs(data_dir: Path) -> List[Path]:
    if data_dir.name.startswith("episode_") and not data_dir.name.endswith("_temp"):
        return [data_dir]
    return sorted(
        p
        for p in data_dir.iterdir()
        if p.is_dir() and p.name.startswith("episode_") and not p.name.endswith("_temp")
    )


def _iter_upload_files(episode_dir: Path) -> List[Path]:
    return sorted(
        (p for p in episode_dir.rglob("*") if p.is_file()),
        key=lambda p: (_is_root_metadata_file(episode_dir, p), str(p.relative_to(episode_dir))),
    )


def _is_root_metadata_file(episode_dir: Path, file_path: Path) -> bool:
    return file_path.relative_to(episode_dir).as_posix() == "metadata.json"


def _upload_episode_name(local_episode_name: str) -> str:
    return local_episode_name.removeprefix("episode_")


def _make_object_key(prefix: str, episode_name: str, relative_path: Path) -> str:
    return str(PurePosixPath(prefix) / episode_name / PurePosixPath(*relative_path.parts))


def _emit(progress: Optional[ProgressCallback], message: str) -> None:
    if progress is not None:
        progress(message)
