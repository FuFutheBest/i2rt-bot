#!/usr/bin/env python3
"""HTTP upload backend for local/LAN pressure testing without DM3 or OSS."""

from __future__ import annotations

import http.client
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional
from urllib.parse import quote, urlparse

from incubator.nedf3.windows_DM3.oss_uploader import UploadProgress, UploadResult, UploadedFile


ProgressCallback = Callable[[str], None]
UploadProgressCallback = Callable[[UploadProgress], None]

DEFAULT_HTTP_UPLOAD_CHUNK_BYTES = 1024 * 1024
DEFAULT_HTTP_UPLOAD_TIMEOUT_SEC = 120.0
HTTP_UPLOAD_LIMIT_MB_S_ENV = "SUPERINFERENCE_HTTP_UPLOAD_LIMIT_MB_S"


class _ProgressTracker:
    def __init__(
        self,
        tasks: List[tuple[Path, str, int]],
        callback: Optional[UploadProgressCallback],
    ) -> None:
        import threading

        self._callback = callback
        self._lock = threading.Lock()
        self.total_files = len(tasks)
        self.total_bytes = sum(size for _path, _key, size in tasks)
        self.completed_files = 0
        self.completed_bytes = 0
        self._inflight_bytes: dict[str, int] = {}

    def start_file(self, object_key: str) -> None:
        with self._lock:
            self._inflight_bytes[object_key] = 0
        self.emit()

    def update_file(self, object_key: str, consumed_bytes: int) -> None:
        with self._lock:
            self._inflight_bytes[object_key] = consumed_bytes
        self.emit()

    def finish_file(self, object_key: str, size_bytes: int) -> None:
        with self._lock:
            self._inflight_bytes.pop(object_key, None)
            self.completed_files += 1
            self.completed_bytes += size_bytes
        self.emit()

    def cancel_file(self, object_key: str) -> None:
        with self._lock:
            self._inflight_bytes.pop(object_key, None)
        self.emit()

    def emit(self) -> None:
        if self._callback is None:
            return
        with self._lock:
            current_bytes = self.completed_bytes + sum(self._inflight_bytes.values())
            progress = UploadProgress(
                completed_files=self.completed_files,
                total_files=self.total_files,
                completed_bytes=self.completed_bytes,
                total_bytes=self.total_bytes,
                current_bytes=current_bytes,
            )
        self._callback(progress)


def upload_episode_to_http_sink(
    episode_path: Path,
    receiver_url: str,
    *,
    progress: Optional[ProgressCallback] = None,
    max_workers: int = 1,
    on_upload_progress: Optional[UploadProgressCallback] = None,
    chunk_bytes: int = DEFAULT_HTTP_UPLOAD_CHUNK_BYTES,
    timeout_sec: float = DEFAULT_HTTP_UPLOAD_TIMEOUT_SEC,
    token: str = "",
    stop_event: Optional[threading.Event] = None,
) -> UploadResult:
    """Upload one finalized episode directory to a small HTTP sink server."""

    episode_path = episode_path.expanduser().resolve()
    if not episode_path.is_dir():
        raise ValueError(f"Episode directory does not exist: {episode_path}")
    if not receiver_url.strip():
        raise ValueError("HTTP receiver URL is required")
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")

    tasks = _collect_episode_files(episode_path)
    if not tasks:
        raise ValueError(f"No files found to upload under: {episode_path}")

    tracker = _ProgressTracker(tasks, on_upload_progress)
    tracker.emit()
    worker_count = min(max_workers, len(tasks))
    _emit(progress, f"HTTP sink upload: {episode_path.name}, files={len(tasks)}, workers={worker_count}")

    if worker_count > 1:
        uploaded = _upload_parallel(
            tasks,
            receiver_url=receiver_url,
            progress=progress,
            tracker=tracker,
            max_workers=worker_count,
            chunk_bytes=chunk_bytes,
            timeout_sec=timeout_sec,
            token=token,
            stop_event=stop_event,
        )
    else:
        uploaded = [
            _upload_one_file(
                task,
                receiver_url=receiver_url,
                progress=progress,
                tracker=tracker,
                chunk_bytes=chunk_bytes,
                timeout_sec=timeout_sec,
                token=token,
                stop_event=stop_event,
            )
            for task in tasks
        ]

    return UploadResult(
        bucket="http-sink",
        endpoint=receiver_url.rstrip("/"),
        prefix=episode_path.name,
        files=uploaded,
    )


def _collect_episode_files(episode_path: Path) -> List[tuple[Path, str, int]]:
    tasks: List[tuple[Path, str, int]] = []
    for file_path in sorted(path for path in episode_path.rglob("*") if path.is_file()):
        relative = file_path.relative_to(episode_path).as_posix()
        object_key = f"{episode_path.name}/{relative}"
        tasks.append((file_path, object_key, file_path.stat().st_size))
    return tasks


def _upload_parallel(
    tasks: List[tuple[Path, str, int]],
    *,
    receiver_url: str,
    progress: Optional[ProgressCallback],
    tracker: _ProgressTracker,
    max_workers: int,
    chunk_bytes: int,
    timeout_sec: float,
    token: str,
    stop_event: Optional[threading.Event],
) -> List[UploadedFile]:
    results: List[Optional[UploadedFile]] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                _upload_one_file,
                task,
                receiver_url=receiver_url,
                progress=progress,
                tracker=tracker,
                chunk_bytes=chunk_bytes,
                timeout_sec=timeout_sec,
                token=token,
                stop_event=stop_event,
            ): index
            for index, task in enumerate(tasks)
        }
        for future in as_completed(future_to_index):
            results[future_to_index[future]] = future.result()
    return [item for item in results if item is not None]


def _upload_one_file(
    task: tuple[Path, str, int],
    *,
    receiver_url: str,
    progress: Optional[ProgressCallback],
    tracker: _ProgressTracker,
    chunk_bytes: int,
    timeout_sec: float,
    token: str,
    stop_event: Optional[threading.Event],
) -> UploadedFile:
    file_path, object_key, size_bytes = task
    parsed = urlparse(receiver_url.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid HTTP receiver URL: {receiver_url}")

    base_path = parsed.path.rstrip("/")
    request_path = f"{base_path}/upload/{quote(object_key, safe='')}"
    if parsed.query:
        request_path = f"{request_path}?{parsed.query}"

    connection_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = connection_cls(parsed.netloc, timeout=timeout_sec)
    upload_limit_mb_s = _read_upload_limit_mb_s()
    tracker.start_file(object_key)
    try:
        _emit(progress, f"HTTP upload {file_path} -> {receiver_url}/upload/{object_key}")
        conn.putrequest("POST", request_path)
        conn.putheader("Content-Length", str(size_bytes))
        conn.putheader("Content-Type", "application/octet-stream")
        conn.putheader("X-Object-Key", object_key)
        conn.putheader("X-File-Name", file_path.name)
        conn.putheader("X-File-Size", str(size_bytes))
        if token:
            conn.putheader("X-Upload-Token", token)
        conn.endheaders()

        sent = 0
        started_at = time.monotonic()
        with file_path.open("rb") as handle:
            while True:
                if stop_event is not None and stop_event.is_set():
                    raise InterruptedError("upload interrupted by shutdown")
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                conn.send(chunk)
                sent += len(chunk)
                tracker.update_file(object_key, sent)
                _throttle_upload(sent, started_at, upload_limit_mb_s, stop_event)

        response = conn.getresponse()
        body = response.read(4096).decode("utf-8", errors="replace")
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"HTTP sink returned {response.status}: {body}")
        tracker.finish_file(object_key, size_bytes)
        return UploadedFile(
            local_path=str(file_path),
            object_key=object_key,
            size_bytes=size_bytes,
            skipped=False,
        )
    except Exception:
        tracker.cancel_file(object_key)
        raise
    finally:
        conn.close()


def _emit(progress: Optional[ProgressCallback], message: str) -> None:
    if progress is not None:
        progress(message)


def _read_upload_limit_mb_s() -> float:
    raw = os.environ.get(HTTP_UPLOAD_LIMIT_MB_S_ENV, "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _throttle_upload(
    sent_bytes: int,
    started_at: float,
    limit_mb_s: float,
    stop_event: Optional[threading.Event],
) -> None:
    if limit_mb_s <= 0:
        return
    target_elapsed = sent_bytes / (limit_mb_s * 1024 * 1024)
    sleep_seconds = target_elapsed - (time.monotonic() - started_at)
    if sleep_seconds <= 0:
        return
    if stop_event is not None:
        stop_event.wait(sleep_seconds)
    else:
        time.sleep(sleep_seconds)

