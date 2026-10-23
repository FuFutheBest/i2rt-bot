#!/usr/bin/env python3
"""Small HTTP receiver for Vive_DM3 upload pressure tests."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Dict
from urllib.parse import unquote, urlparse


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SinkConfig:
    output_dir: Path
    discard: bool = False
    limit_mbps: float = 0.0
    fail_rate: float = 0.0
    disconnect_rate: float = 0.0
    chunk_bytes: int = 1024 * 1024
    max_file_bytes: int = 32 * 1024 * 1024 * 1024
    min_free_bytes: int = 10 * 1024 * 1024 * 1024
    max_active_requests: int = 4
    token: str = ""


@dataclass
class SinkStats:
    started_at: str = field(default_factory=utc_now_iso)
    total_requests: int = 0
    active_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    disconnected_requests: int = 0
    files_received: int = 0
    bytes_received: int = 0
    last_error: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "started_at": self.started_at,
                "updated_at": utc_now_iso(),
                "total_requests": self.total_requests,
                "active_requests": self.active_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "disconnected_requests": self.disconnected_requests,
                "files_received": self.files_received,
                "bytes_received": self.bytes_received,
                "last_error": self.last_error,
            }


class UploadSinkServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], config: SinkConfig) -> None:
        super().__init__(server_address, UploadSinkHandler)
        self.config = config
        self.stats = SinkStats()
        self.manifest_path = config.output_dir / "uploads.jsonl"
        self.manifest_lock = threading.Lock()


class UploadSinkHandler(BaseHTTPRequestHandler):
    server: UploadSinkServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.log_date_time_string()} {self.client_address[0]} {fmt % args}", flush=True)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            if not self._authorized():
                self._send_json(401, {"error": "unauthorized"})
                return
            self._send_json(200, {"ok": True, "stats": self.server.stats.snapshot()})
            return
        if parsed.path == "/stats":
            if not self._authorized():
                self._send_json(401, {"error": "unauthorized"})
                return
            self._send_json(200, self.server.stats.snapshot())
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        prefix = "/upload/"
        if not parsed.path.startswith(prefix):
            self._send_json(404, {"error": "not found"})
            return
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return

        object_key = unquote(parsed.path[len(prefix) :])
        try:
            dest_path = _safe_destination(self.server.config.output_dir, object_key)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        content_length = self.headers.get("Content-Length")
        if content_length is None:
            self._send_json(411, {"error": "Content-Length is required"})
            return
        try:
            expected_bytes = int(content_length)
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length"})
            return
        if expected_bytes < 0:
            self._send_json(400, {"error": "Content-Length must be non-negative"})
            return

        config = self.server.config
        stats = self.server.stats
        if expected_bytes > config.max_file_bytes:
            self._send_json(
                413,
                {
                    "error": "file too large",
                    "max_file_bytes": config.max_file_bytes,
                    "content_length": expected_bytes,
                },
            )
            return
        if not config.discard and not _has_enough_free_space(
            config.output_dir,
            required_bytes=expected_bytes + config.min_free_bytes,
        ):
            self._send_json(
                507,
                {
                    "error": "not enough free disk space",
                    "required_payload_bytes": expected_bytes,
                    "min_free_bytes": config.min_free_bytes,
                },
            )
            return

        should_disconnect = random.random() < config.disconnect_rate
        should_fail = random.random() < config.fail_rate

        too_many_active = False
        with stats.lock:
            if stats.active_requests >= config.max_active_requests:
                stats.failed_requests += 1
                stats.last_error = f"too many active requests: {stats.active_requests}"
                too_many_active = True
            else:
                stats.total_requests += 1
                stats.active_requests += 1
        if too_many_active:
            self._send_json(
                429,
                {
                    "error": "too many active requests",
                    "max_active_requests": config.max_active_requests,
                },
            )
            return

        if should_disconnect:
            with stats.lock:
                stats.active_requests -= 1
                stats.disconnected_requests += 1
                stats.last_error = f"planned disconnect: {object_key}"
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
            return

        received = 0
        temp_path = dest_path.with_name(dest_path.name + ".part")
        started = time.monotonic()
        writer = None
        try:
            if not config.discard:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                writer = temp_path.open("wb")

            remaining = expected_bytes
            while remaining > 0:
                chunk = self.rfile.read(min(config.chunk_bytes, remaining))
                if not chunk:
                    raise ConnectionError("client disconnected before full body arrived")
                remaining -= len(chunk)
                received += len(chunk)
                if writer is not None:
                    writer.write(chunk)
                self._apply_rate_limit(received, started)

            if writer is not None:
                writer.close()
                writer = None

            if should_fail:
                if temp_path.exists():
                    temp_path.unlink()
                with stats.lock:
                    stats.failed_requests += 1
                    stats.last_error = f"planned HTTP 500: {object_key}"
                self._send_json(500, {"ok": False, "error": "planned failure", "object_key": object_key})
                return

            if not config.discard:
                temp_path.replace(dest_path)
            self._append_manifest(object_key, expected_bytes, received, dest_path)
            with stats.lock:
                stats.successful_requests += 1
                stats.files_received += 1
                stats.bytes_received += received
            self._send_json(
                200,
                {
                    "ok": True,
                    "object_key": object_key,
                    "bytes_received": received,
                    "discard": config.discard,
                },
            )
        except Exception as exc:
            if writer is not None:
                writer.close()
            if temp_path.exists():
                temp_path.unlink()
            with stats.lock:
                stats.failed_requests += 1
                stats.last_error = str(exc)
            self._send_json(500, {"ok": False, "error": str(exc), "object_key": object_key})
        finally:
            with stats.lock:
                stats.active_requests -= 1

    def _apply_rate_limit(self, received: int, started: float) -> None:
        limit_mbps = self.server.config.limit_mbps
        if limit_mbps <= 0:
            return
        target_elapsed = received / (limit_mbps * 1024 * 1024)
        actual_elapsed = time.monotonic() - started
        if target_elapsed > actual_elapsed:
            time.sleep(target_elapsed - actual_elapsed)

    def _append_manifest(self, object_key: str, expected_bytes: int, received: int, dest_path: Path) -> None:
        payload = {
            "time": utc_now_iso(),
            "client": self.client_address[0],
            "object_key": object_key,
            "expected_bytes": expected_bytes,
            "bytes_received": received,
            "stored_path": "" if self.server.config.discard else str(dest_path),
        }
        self.server.config.output_dir.mkdir(parents=True, exist_ok=True)
        with self.server.manifest_lock:
            with self.server.manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _authorized(self) -> bool:
        token = self.server.config.token
        if not token:
            return True
        return self.headers.get("X-Upload-Token", "") == token

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def _safe_destination(root: Path, object_key: str) -> Path:
    if "\\" in object_key or ":" in object_key:
        raise ValueError(f"unsafe object key: {object_key}")
    posix = PurePosixPath(object_key)
    if posix.is_absolute():
        raise ValueError("object key must be relative")
    if not posix.parts or any(part in {"", ".", ".."} for part in posix.parts):
        raise ValueError(f"unsafe object key: {object_key}")
    root_resolved = root.resolve()
    dest = root_resolved.joinpath(*posix.parts).resolve()
    if dest != root_resolved and root_resolved not in dest.parents:
        raise ValueError(f"unsafe object key: {object_key}")
    return dest


def _has_enough_free_space(path: Path, *, required_bytes: int) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free >= required_bytes


def _parse_size_bytes(raw: str) -> int:
    text = str(raw).strip().lower()
    units = {
        "b": 1,
        "kb": 1024,
        "mb": 1024**2,
        "gb": 1024**3,
        "tb": 1024**4,
    }
    for suffix, multiplier in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)].strip()) * multiplier)
    return int(float(text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive Vive_DM3 pressure-test uploads over HTTP.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host, default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=18080, help="Bind port, default: 18080")
    parser.add_argument("--output-dir", default="upload_sink_received", help="Directory for received files and uploads.jsonl")
    parser.add_argument("--discard", action="store_true", help="Read uploads but do not store file payloads")
    parser.add_argument("--limit-mbps", type=float, default=0.0, help="Per-request read limit; 0 means unlimited")
    parser.add_argument("--fail-rate", type=float, default=0.0, help="Probability of returning HTTP 500 after reading a file")
    parser.add_argument("--disconnect-rate", type=float, default=0.0, help="Probability of closing a request before reading")
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024, help="Read chunk size")
    parser.add_argument("--max-file-size", default="32GB", help="Reject a single file above this size, default: 32GB")
    parser.add_argument("--min-free-space", default="10GB", help="Keep at least this much free disk space, default: 10GB")
    parser.add_argument("--max-active-requests", type=int, default=4, help="Maximum concurrent upload requests")
    parser.add_argument("--token", default="", help="Optional shared token required in X-Upload-Token")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SinkConfig(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        discard=bool(args.discard),
        limit_mbps=max(0.0, float(args.limit_mbps)),
        fail_rate=min(1.0, max(0.0, float(args.fail_rate))),
        disconnect_rate=min(1.0, max(0.0, float(args.disconnect_rate))),
        chunk_bytes=max(4096, int(args.chunk_bytes)),
        max_file_bytes=max(1, _parse_size_bytes(args.max_file_size)),
        min_free_bytes=max(0, _parse_size_bytes(args.min_free_space)),
        max_active_requests=max(1, int(args.max_active_requests)),
        token=str(args.token or ""),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    server = UploadSinkServer((args.host, args.port), config)
    print(
        json.dumps(
            {
                "message": "Vive_DM3 upload sink listening",
                "url": f"http://{args.host}:{args.port}",
                "health": f"http://{args.host}:{args.port}/health",
                "output_dir": str(config.output_dir),
                "discard": config.discard,
                "limit_mbps": config.limit_mbps,
                "fail_rate": config.fail_rate,
                "disconnect_rate": config.disconnect_rate,
                "max_file_bytes": config.max_file_bytes,
                "min_free_bytes": config.min_free_bytes,
                "max_active_requests": config.max_active_requests,
                "token_required": bool(config.token),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping upload sink...", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

