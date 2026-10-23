#!/usr/bin/env python3
"""
Aliyun OSS uploader for DM3 Windows local collection data.

Author: yichen Xie
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional

from utils.logger_config import logger


ProgressCallback = Callable[[str], None]
TokenProvider = Callable[[], Dict[str, Any]]

# Refresh the STS token this many seconds before its declared expiration, so a
# long-running put never starts with an about-to-expire token.
_TOKEN_REFRESH_MARGIN_S = 120.0


@dataclass
class UploadProgress:
    """A snapshot of overall upload progress, suitable for driving a progress bar.

    ``uploaded_bytes`` includes the bytes of files skipped (already on OSS) plus
    the bytes streamed so far for the file currently uploading, so the ratio
    ``uploaded_bytes / total_bytes`` advances smoothly even within a large file.
    """

    uploaded_bytes: int
    total_bytes: int
    uploaded_files: int
    total_files: int
    current_file: str


ProgressReporter = Callable[[UploadProgress], None]


@dataclass
class UploadedFile:
    """One local file and its OSS object key (may be skipped if already present on OSS)."""

    local_path: str
    object_key: str
    size_bytes: int


@dataclass
class UploadResult:
    """Summary of one upload run."""

    bucket: str
    endpoint: str
    prefix: str
    files: List[UploadedFile]


def upload_episode_data_dir(
    data_dir: Path,
    sts_token: Dict[str, Any],
    *,
    progress: Optional[ProgressCallback] = None,
    on_progress: Optional[ProgressReporter] = None,
    token_provider: Optional[TokenProvider] = None,
    skip_existing: bool = True,
) -> UploadResult:
    """
    Upload a local task directory or a single episode directory to OSS.

    Object layout:
        {sts.dir}/{episode_uuid}/{relative_file_path}

    Local episode directories are bare UUIDs (no ``episode_`` prefix); any
    legacy ``episode_`` prefix is stripped from the OSS object key.

    If ``data_dir`` itself is an episode directory, only that episode is uploaded.
    Otherwise finalized child episode directories (containing ``metadata.json``)
    are uploaded. Temporary directories ending in ``_temp`` are skipped.

    STS tokens are short-lived, so a long upload can outlive a single token. When
    ``token_provider`` is given the session refreshes the token before it expires
    and also retries once with a fresh token if OSS reports the token expired.

    When ``skip_existing`` is True (default), files already present on OSS with a
    matching size are skipped, so an interrupted run can be re-run and only the
    remaining files are uploaded.

    ``progress`` receives human-readable status lines; ``on_progress`` receives an
    :class:`UploadProgress` snapshot after every file (and during a file as bytes
    stream), so callers can drive a progress bar.
    """
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist or is not a directory: {data_dir}")

    session = _OssSession(sts_token, token_provider=token_provider, progress=progress)

    episode_dirs = _discover_episode_dirs(data_dir)
    if not episode_dirs:
        raise ValueError(f"No finalized episode directories found under: {data_dir}")

    # Flatten everything first so we know total file/byte counts up front; the
    # progress bar needs a denominator before the first byte is sent.
    plan: List[tuple[Path, str, int]] = []
    for episode_dir in episode_dirs:
        upload_episode_name = _upload_episode_name(episode_dir.name)
        for file_path in _iter_upload_files(episode_dir):
            relative_path = file_path.relative_to(episode_dir)
            object_key = _make_object_key(session.prefix, upload_episode_name, relative_path)
            plan.append((file_path, object_key, file_path.stat().st_size))

    total_files = len(plan)
    total_bytes = sum(size for _, _, size in plan)
    done_bytes = 0
    done_files = 0
    _report(on_progress, UploadProgress(0, total_bytes, 0, total_files, ""))

    uploaded_files: List[UploadedFile] = []
    skipped = 0
    for file_path, object_key, local_size in plan:
        if skip_existing and session.remote_size(object_key) == local_size:
            skipped += 1
            _emit(progress, f"Skip (already on OSS) {file_path}")
        else:
            _emit(
                progress,
                f"Uploading {file_path} -> oss://{session.bucket_name}/{object_key}",
            )

            def _byte_cb(consumed: int, _total: int, base: int = done_bytes,
                         fp: str = str(file_path)) -> None:
                # oss2 reports (bytes_consumed, file_total); fold it into the
                # overall byte counter via the bytes already finished (``base``).
                _report(
                    on_progress,
                    UploadProgress(base + int(consumed), total_bytes, done_files, total_files, fp),
                )

            session.put_file(object_key, file_path, progress_callback=_byte_cb)

        done_bytes += local_size
        done_files += 1
        _report(
            on_progress,
            UploadProgress(done_bytes, total_bytes, done_files, total_files, str(file_path)),
        )
        uploaded_files.append(
            UploadedFile(
                local_path=str(file_path),
                object_key=object_key,
                size_bytes=local_size,
            )
        )

    logger.info(
        f"Uploaded {len(uploaded_files) - skipped} files (skipped {skipped} already "
        f"on OSS) from {data_dir} to oss://{session.bucket_name}/{session.prefix} "
        f"(local episode_<uuid> -> OSS <uuid>)"
    )
    return UploadResult(
        bucket=session.bucket_name,
        endpoint=session.endpoint,
        prefix=session.prefix,
        files=uploaded_files,
    )


class _OssSession:
    """Holds the OSS bucket bound to the current STS token and transparently
    refreshes that token (proactively near expiry, and reactively when OSS
    reports ``SecurityTokenExpired``) using an optional ``token_provider``.
    """

    def __init__(
        self,
        sts_token: Dict[str, Any],
        *,
        token_provider: Optional[TokenProvider] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        self._oss2 = _import_oss2()
        self._token_provider = token_provider
        self._progress = progress
        self._expires_at: Optional[float] = None
        self.bucket_name: str = ""
        self.endpoint: str = ""
        self.prefix: str = ""
        self.bucket: Any = None
        self._apply_token(sts_token)

    def _apply_token(self, sts_token: Dict[str, Any]) -> None:
        access_key_id = _required_str(sts_token, "access_key_id")
        access_key_secret = _required_str(sts_token, "access_key_secret")
        security_token = _required_str(sts_token, "security_token")
        self.bucket_name = _required_str(sts_token, "bucket")
        self.endpoint = _required_str(sts_token, "endpoint")
        self.prefix = _normalize_prefix(_required_str(sts_token, "dir"))
        self._expires_at = _parse_expiration(sts_token)

        auth = self._oss2.StsAuth(access_key_id, access_key_secret, security_token)
        self.bucket = self._oss2.Bucket(auth, _normalize_endpoint(self.endpoint), self.bucket_name)

    def _refresh(self) -> None:
        if self._token_provider is None:
            return
        _emit(self._progress, "Refreshing OSS STS token...")
        self._apply_token(self._token_provider())

    def _refresh_if_near_expiry(self) -> None:
        if self._token_provider is None or self._expires_at is None:
            return
        if time.time() >= self._expires_at - _TOKEN_REFRESH_MARGIN_S:
            self._refresh()

    def _call_with_refresh(self, func: Callable[[], Any]) -> Any:
        self._refresh_if_near_expiry()
        try:
            return func()
        except self._oss2.exceptions.OssError as exc:
            if self._token_provider is not None and _is_token_expired(exc):
                _emit(self._progress, "OSS STS token expired; refreshing and retrying...")
                self._refresh()
                return func()
            raise

    def put_file(
        self,
        object_key: str,
        file_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self._call_with_refresh(
            lambda: self.bucket.put_object_from_file(
                object_key, str(file_path), progress_callback=progress_callback
            )
        )

    def remote_size(self, object_key: str) -> Optional[int]:
        """Return the OSS object's size in bytes, or None if it does not exist."""
        def _head() -> Optional[int]:
            try:
                return int(self.bucket.head_object(object_key).content_length)
            except self._oss2.exceptions.NoSuchKey:
                return None
            except self._oss2.exceptions.NotFound:
                return None

        return self._call_with_refresh(_head)


def _import_oss2() -> Any:
    try:
        import oss2
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'oss2'. Install it with: pip install oss2"
        ) from exc
    return oss2


def _required_str(data: Dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not value:
        raise ValueError(f"STS token response missing required field: {key}")
    return str(value)


def _parse_expiration(sts_token: Dict[str, Any]) -> Optional[float]:
    """Best-effort parse of the STS token expiration into an epoch timestamp.

    Accepts common shapes (``expiration``/``Expiration``/``expire_at``) as either
    an ISO-8601 string (optionally ``Z``-suffixed) or an epoch number. Returns
    None when no usable expiration is present, in which case the session relies on
    reactive refresh-on-error instead of proactive refresh.
    """
    for key in ("expiration", "Expiration", "expire_at", "expireAt", "expires"):
        raw = sts_token.get(key)
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            return float(raw)
        try:
            text = str(raw).strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except (ValueError, TypeError):
            continue
    return None


def _is_token_expired(exc: Any) -> bool:
    """True when an oss2 error is an expired/invalid STS token (so a fresh token
    would let the request succeed)."""
    code = str(getattr(exc, "code", "") or "")
    return code in ("SecurityTokenExpired", "InvalidSecurityToken", "InvalidAccessKeyId")


def _normalize_endpoint(endpoint: str) -> str:
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    return f"https://{endpoint}"


def _normalize_prefix(prefix: str) -> str:
    return prefix.strip().strip("/")


def _is_finalized_episode_dir(path: Path) -> bool:
    """A finalized episode dir: not a *_temp dir, and contains metadata.json.

    Names are bare UUIDs (no ``episode_`` prefix), so we identify episodes by
    the presence of the finalized ``metadata.json`` marker instead of by name.
    """
    return (
        path.is_dir()
        and not path.name.endswith("_temp")
        and (path / "metadata.json").is_file()
    )


def _discover_episode_dirs(data_dir: Path) -> List[Path]:
    if _is_finalized_episode_dir(data_dir):
        return [data_dir]
    return sorted(p for p in data_dir.iterdir() if _is_finalized_episode_dir(p))


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


def _report(on_progress: Optional[ProgressReporter], snapshot: UploadProgress) -> None:
    if on_progress is not None:
        on_progress(snapshot)
