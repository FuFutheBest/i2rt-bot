#!/usr/bin/env python3
"""
OSS multipart / resumable upload tuning (not exposed in GUI).

Adjust defaults here or override via environment variables:
  OSS_UPLOAD_MULTIPART_THRESHOLD_BYTES  (default: 8388608 = 8 MiB)
  OSS_UPLOAD_PART_SIZE_BYTES            (default: 8388608 = 8 MiB)
  OSS_UPLOAD_NUM_THREADS                (default: 4)
  OSS_UPLOAD_MAX_INFLIGHT_REQUESTS      (default: 6)

Default part_size favors finer UI progress updates for NEDF3 episode uploads.

Author: Cursor Agent
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_OSS_MIN_PART_SIZE_BYTES = 100 * 1024
_DEFAULT_PART_SIZE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class OssMultipartUploadConfig:
    """Tuning for oss2.resumable_upload on large episode files (e.g. data.mcap)."""

    multipart_threshold_bytes: int = 8 * 1024 * 1024
    part_size: int = _DEFAULT_PART_SIZE_BYTES
    num_threads: int = 4
    max_inflight_requests: int = 6

    def __post_init__(self) -> None:
        if self.multipart_threshold_bytes < 1:
            raise ValueError("multipart_threshold_bytes must be >= 1")
        if self.part_size < _OSS_MIN_PART_SIZE_BYTES:
            raise ValueError(
                f"part_size must be >= {_OSS_MIN_PART_SIZE_BYTES} bytes (OSS minimum)"
            )
        if self.num_threads < 1:
            raise ValueError("num_threads must be >= 1")
        if self.max_inflight_requests < 1:
            raise ValueError("max_inflight_requests must be >= 1")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def load_oss_multipart_upload_config() -> OssMultipartUploadConfig:
    """Load upload tuning from defaults and optional environment overrides."""
    return OssMultipartUploadConfig(
        multipart_threshold_bytes=_env_int(
            "OSS_UPLOAD_MULTIPART_THRESHOLD_BYTES",
            8 * 1024 * 1024,
        ),
        part_size=_env_int("OSS_UPLOAD_PART_SIZE_BYTES", _DEFAULT_PART_SIZE_BYTES),
        num_threads=_env_int("OSS_UPLOAD_NUM_THREADS", 4),
        max_inflight_requests=_env_int("OSS_UPLOAD_MAX_INFLIGHT_REQUESTS", 6),
    )


DEFAULT_OSS_MULTIPART_UPLOAD_CONFIG = load_oss_multipart_upload_config()

