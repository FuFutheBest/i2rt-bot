"""External, bounded cache primitives for NEDF3 recording replay.

Complete cache generations are immutable directories published with one atomic
rename.  The cache never passes the source episode to the private SDK and never
writes inside the recording tree.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Optional

try:  # Linux production path; thread locking remains as a portable fallback.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX only.
    fcntl = None  # type: ignore[assignment]


CACHE_SCHEMA_VERSION = 3
MANIFEST_SCHEMA_VERSION = 1
URDF_REPLAY_SCHEMA_VERSION = 2
DEFAULT_CACHE_TTL_S = 30 * 24 * 60 * 60
# The cap charges generation payloads/manifests plus quarantine, build state,
# access markers and stale work. Tiny lock files, index JSON and filesystem
# directory blocks remain intentionally approximate rather than recursively
# restatting the whole cache on every cleanup.
DEFAULT_CACHE_MAX_BYTES = 20 * 1024 * 1024 * 1024
DEFAULT_ACCESS_TOUCH_S = 60.0
DEFAULT_LOCK_WAIT_S = 0.15
DEFAULT_SHADOW_SOURCE_MAX_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_SHADOW_ROOT = Path("/tmp/superinference_replay_shadows")
STALE_WORK_TTL_S = 10 * 60
MAX_SOURCE_FILES = 512
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_METADATA_HASH_BYTES = 8 * 1024 * 1024
REQUIRED_GENERATION_FILES = frozenset({"summary.json", "timeline.npz"})
OPTIONAL_VIDEO_FILES = frozenset(
    {"video_head.mp4", "video_left.mp4", "video_right.mp4"}
)
ALLOWED_GENERATION_FILES = REQUIRED_GENERATION_FILES | OPTIONAL_VIDEO_FILES


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _safe_json(path: Path, *, max_bytes: int) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError:
        return {}
    if path.is_symlink() or not path.is_file() or not 0 < info.st_size <= max_bytes:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_private_directory(path.parent)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_small_file(path: Path, size: int) -> str:
    if size > MAX_METADATA_HASH_BYTES:
        return "oversized"
    return _sha256_file(path)


@dataclass(frozen=True)
class SourceIdentity:
    fingerprint: str
    artifacts: tuple[tuple[str, int, int, int, int, Optional[str]], ...]

    @property
    def source_bytes(self) -> int:
        return sum(item[1] for item in self.artifacts)


@dataclass(frozen=True)
class CacheEntry:
    directory: Path
    source_fingerprint: str
    profile_fingerprint: str

    @property
    def summary_path(self) -> Path:
        return self.directory / "summary.json"

    @property
    def timeline_path(self) -> Path:
        return self.directory / "timeline.npz"

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    def video_path(self, role: str) -> Path:
        if role not in {"head", "left", "right"}:
            raise ValueError("invalid_recording_video_role")
        return self.directory / f"video_{role}.mp4"


@dataclass(frozen=True)
class CacheLoad:
    status: str
    inspection: Optional[dict[str, Any]] = None
    total_bytes: int = 0


class RecordingReplayCache:
    """Filesystem cache with bounded identities, generations and eviction."""

    def __init__(
        self,
        root: Path,
        *,
        ttl_s: float = DEFAULT_CACHE_TTL_S,
        max_bytes: int = DEFAULT_CACHE_MAX_BYTES,
        access_touch_s: float = DEFAULT_ACCESS_TOUCH_S,
        shadow_source_max_bytes: int = DEFAULT_SHADOW_SOURCE_MAX_BYTES,
        shadow_root: Path = DEFAULT_SHADOW_ROOT,
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=False) / f"v{CACHE_SCHEMA_VERSION}"
        self.entries_root = self.root / "entries"
        self.roots_root = self.root / "roots"
        self.locks_root = self.root / "locks"
        shadow_base = Path(shadow_root).expanduser().resolve(strict=False)
        if any("." in part for part in shadow_base.parts if part not in {"/", ""}):
            raise ValueError("recording_shadow_root_contains_dot")
        cache_namespace = hashlib.sha256(str(self.root).encode("utf-8")).hexdigest()[:24]
        self.shadows_root = shadow_base / cache_namespace
        self.quarantine_root = self.root / "quarantine"
        self.states_root = self.root / "states"
        self.access_root = self.root / "access"
        self.ttl_s = max(0.0, float(ttl_s))
        self.max_bytes = max(1, int(max_bytes))
        self.access_touch_s = max(1.0, float(access_touch_s))
        self.shadow_source_max_bytes = max(1, int(shadow_source_max_bytes))
        self._thread_locks_guard = threading.Lock()
        self._thread_locks: dict[str, threading.Lock] = {}
        self._verified_guard = threading.Lock()
        self._verified: set[tuple[str, int]] = set()
        for directory in (
            self.root,
            self.entries_root,
            self.roots_root,
            self.locks_root,
            self.quarantine_root,
            self.states_root,
            self.access_root,
        ):
            _ensure_private_directory(directory)

    def source_identity(self, episode: Path) -> SourceIdentity:
        episode = Path(episode).resolve(strict=True)
        candidates: list[Path] = [
            episode / "metadata.json",
            episode / "mcap_index.json",
        ]
        data = episode / "data"
        if data.is_dir() and not data.is_symlink():
            candidates.append(data / "mcap_index.json")
            try:
                children = sorted(data.iterdir(), key=lambda item: item.name)
            except OSError as exc:
                raise ValueError("recording_data_unreadable") from exc
            candidates.extend(child for child in children if child.suffix.casefold() == ".mcap")
        candidates.extend(sorted(episode.glob("*.mcap"), key=lambda item: item.name))

        unique: dict[str, Path] = {}
        for candidate in candidates:
            try:
                unique[candidate.relative_to(episode).as_posix()] = candidate
            except ValueError:
                continue
        if len(unique) > MAX_SOURCE_FILES:
            raise ValueError("recording_source_file_limit")

        artifacts: list[tuple[str, int, int, int, int, Optional[str]]] = []
        for relative, candidate in sorted(unique.items()):
            try:
                info = candidate.lstat()
            except FileNotFoundError:
                if relative == "metadata.json":
                    raise ValueError("recording_metadata_missing")
                continue
            except OSError as exc:
                raise ValueError("recording_source_unreadable") from exc
            if candidate.is_symlink() or not candidate.is_file():
                if relative == "metadata.json":
                    raise ValueError("recording_metadata_unsafe")
                continue
            content_hash: Optional[str] = None
            if relative in {"metadata.json", "mcap_index.json", "data/mcap_index.json"}:
                try:
                    content_hash = _hash_small_file(candidate, int(info.st_size))
                except OSError as exc:
                    raise ValueError("recording_source_unreadable") from exc
            artifacts.append(
                (
                    relative,
                    int(info.st_size),
                    int(info.st_mtime_ns),
                    int(info.st_dev),
                    int(info.st_ino),
                    content_hash,
                )
            )
        if not any(artifact[0] == "metadata.json" for artifact in artifacts):
            raise ValueError("recording_metadata_missing")
        encoded = json.dumps(
            [CACHE_SCHEMA_VERSION, artifacts], separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return SourceIdentity(hashlib.sha256(encoded).hexdigest(), tuple(artifacts))

    @staticmethod
    def profile_fingerprint(
        camera_roles: Mapping[str, Any],
        *,
        urdf_schema_version: int = URDF_REPLAY_SCHEMA_VERSION,
    ) -> str:
        normalized = {
            str(key): value if isinstance(value, (str, int, float, bool)) or value is None else str(value)
            for key, value in sorted(camera_roles.items())
        }
        encoded = json.dumps(
            {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "urdf_schema_version": int(urdf_schema_version),
                "camera_roles": normalized,
            },
            separators=(",", ":"),
            ensure_ascii=True,
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _episode_key(episode: Path) -> str:
        return hashlib.sha256(str(Path(episode).resolve(strict=False)).encode("utf-8")).hexdigest()

    def entry(
        self,
        episode: Path,
        source_fingerprint: str,
        profile_fingerprint: str,
    ) -> CacheEntry:
        episode_key = self._episode_key(episode)
        return CacheEntry(
            self.entries_root
            / episode_key[:2]
            / episode_key
            / source_fingerprint
            / profile_fingerprint,
            source_fingerprint,
            profile_fingerprint,
        )

    def _thread_lock(self, key: str) -> threading.Lock:
        with self._thread_locks_guard:
            return self._thread_locks.setdefault(key, threading.Lock())

    @contextmanager
    def lock(self, key: str, *, timeout_s: Optional[float] = None) -> Iterator[bool]:
        lock_key = hashlib.sha256(key.encode("utf-8")).hexdigest()
        thread_lock = self._thread_lock(lock_key)
        if timeout_s is None:
            thread_acquired = thread_lock.acquire()
        else:
            thread_acquired = thread_lock.acquire(timeout=max(0.0, timeout_s))
        if not thread_acquired:
            yield False
            return
        handle = None
        file_acquired = False
        try:
            lock_path = self.locks_root / f"{lock_key}.lock"
            handle = lock_path.open("a+b")
            try:
                os.chmod(lock_path, 0o600)
            except OSError:
                pass
            if fcntl is None:
                file_acquired = True
            elif timeout_s is None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                file_acquired = True
            else:
                deadline = time.monotonic() + max(0.0, timeout_s)
                while True:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        file_acquired = True
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            break
                        time.sleep(0.01)
            yield file_acquired
        finally:
            if file_acquired and fcntl is not None and handle is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            if handle is not None:
                handle.close()
            thread_lock.release()

    @contextmanager
    def entry_lock(
        self,
        entry: CacheEntry,
        *,
        timeout_s: Optional[float] = DEFAULT_LOCK_WAIT_S,
    ) -> Iterator[bool]:
        with self.lock(f"entry:{entry.directory}", timeout_s=timeout_s) as acquired:
            yield acquired

    def _state_path(self, entry: CacheEntry) -> Path:
        key = hashlib.sha256(str(entry.directory).encode("utf-8")).hexdigest()
        return self.states_root / f"{key}.json"

    def _access_path(self, entry: CacheEntry) -> Path:
        key = hashlib.sha256(str(entry.directory).encode("utf-8")).hexdigest()
        return self.access_root / key

    def _touch_access(self, entry: CacheEntry) -> None:
        path = self._access_path(entry)
        try:
            info = path.stat()
            if time.time() - info.st_mtime < self.access_touch_s:
                return
            os.utime(path, None)
        except FileNotFoundError:
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(descriptor)
            except FileExistsError:
                pass
            except OSError:
                pass
        except OSError:
            pass

    def _last_access_ns(self, entry: CacheEntry, fallback_ns: int) -> int:
        try:
            return int(self._access_path(entry).stat().st_mtime_ns)
        except OSError:
            return fallback_ns

    def read_build_state(self, entry: CacheEntry) -> dict[str, Any]:
        value = _safe_json(self._state_path(entry), max_bytes=MAX_MANIFEST_BYTES)
        if (
            value.get("schema_version") != CACHE_SCHEMA_VERSION
            or value.get("source_fingerprint") != entry.source_fingerprint
            or value.get("profile_fingerprint") != entry.profile_fingerprint
        ):
            return {}
        return value

    def write_build_state(
        self,
        entry: CacheEntry,
        status: str,
        *,
        attempts: int = 0,
        retry_at_ns: int = 0,
        code: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        _atomic_json(
            self._state_path(entry),
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "source_fingerprint": entry.source_fingerprint,
                "profile_fingerprint": entry.profile_fingerprint,
                "status": status,
                "attempts": max(0, int(attempts)),
                "retry_at_ns": max(0, int(retry_at_ns)),
                "code": code,
                "error": error[:300] if isinstance(error, str) else None,
                "updated_at_ns": time.time_ns(),
            },
        )

    @contextmanager
    def shadow_episode(
        self,
        episode: Path,
        identity: SourceIdentity,
    ) -> Iterator[Path]:
        """Create a bounded hardlink-only SDK view outside the source tree."""
        if identity.source_bytes > self.shadow_source_max_bytes:
            raise ValueError("recording_shadow_source_limit")
        episode = Path(episode).resolve(strict=True)
        safe_episode = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in episode.name
        )[:120]
        with self.lock("shadow-namespace", timeout_s=None) as acquired:
            if not acquired:  # pragma: no cover - blocking lock always acquires.
                raise ValueError("recording_shadow_namespace_unavailable")
            _ensure_private_directory(self.shadows_root)
            raw = tempfile.mkdtemp(
                prefix=f"shadow_{safe_episode}_",
                dir=self.shadows_root,
            )
        shadow = Path(raw)
        try:
            os.chmod(shadow, 0o700)
            artifact_names = {artifact[0] for artifact in identity.artifacts}
            for relative, _size, _mtime, _dev, _ino, _digest in identity.artifacts:
                source = episode / relative
                destination = shadow / relative
                _ensure_private_directory(destination.parent)
                if relative.casefold().endswith(".mcap"):
                    try:
                        os.link(source, destination, follow_symlinks=False)
                    except OSError as exc:
                        raise ValueError("recording_shadow_hardlink_unavailable") from exc
                else:
                    shutil.copyfile(source, destination, follow_symlinks=False)
                    try:
                        os.chmod(destination, 0o400)
                    except OSError:
                        pass
            if "mcap_index.json" in artifact_names:
                # Some writers accept root-layout episodes in
                # parts of the stack, while its chunk reader still hard-codes
                # data/mcap_index.json.  Preserve the source-relative view and
                # add a hardlink-only compatibility view inside the shadow.
                compatibility = shadow / "data"
                _ensure_private_directory(compatibility)
                compatibility_index = compatibility / "mcap_index.json"
                if not compatibility_index.exists():
                    shutil.copyfile(
                        episode / "mcap_index.json",
                        compatibility_index,
                        follow_symlinks=False,
                    )
                    os.chmod(compatibility_index, 0o400)
                for relative in sorted(artifact_names):
                    relative_path = Path(relative)
                    if relative_path.parent != Path(".") or relative_path.suffix.casefold() != ".mcap":
                        continue
                    compatibility_mcap = compatibility / relative_path.name
                    if compatibility_mcap.exists():
                        continue
                    try:
                        os.link(
                            episode / relative,
                            compatibility_mcap,
                            follow_symlinks=False,
                        )
                    except OSError as exc:
                        raise ValueError("recording_shadow_hardlink_unavailable") from exc
            yield shadow
        finally:
            with self.lock("shadow-namespace", timeout_s=None) as acquired:
                if acquired:
                    shutil.rmtree(shadow, ignore_errors=True)
                    try:
                        self.shadows_root.rmdir()
                    except OSError:
                        pass

    def prepare_staging(self, entry: CacheEntry) -> tuple[Path, Path]:
        parent = entry.directory.parent
        _ensure_private_directory(parent)
        staging = Path(tempfile.mkdtemp(prefix=f".{entry.profile_fingerprint}.build.", dir=parent))
        os.chmod(staging, 0o700)
        timeline = staging / "timeline.npz"
        descriptor = os.open(timeline, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        return staging, timeline

    @staticmethod
    def discard_staging(staging: Path) -> None:
        shutil.rmtree(staging, ignore_errors=True)

    def publish(
        self,
        entry: CacheEntry,
        *,
        inspection: Mapping[str, Any],
        staging: Path,
    ) -> None:
        staging = Path(staging)
        timeline = staging / "timeline.npz"
        try:
            timeline_info = timeline.lstat()
        except OSError as exc:
            raise ValueError("recording_timeline_missing") from exc
        if timeline.is_symlink() or not timeline.is_file() or timeline_info.st_size <= 0:
            raise ValueError("recording_timeline_invalid")
        if timeline_info.st_size > self.max_bytes:
            raise ValueError("recording_timeline_exceeds_cache_limit")

        summary = staging / "summary.json"
        _atomic_json(
            summary,
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "source_fingerprint": entry.source_fingerprint,
                "profile_fingerprint": entry.profile_fingerprint,
                "inspection": dict(inspection),
            },
        )
        try:
            staged_names = {
                child.name
                for child in staging.iterdir()
                if child.name != "manifest.json"
            }
        except OSError as exc:
            raise ValueError("recording_staging_unreadable") from exc
        if not REQUIRED_GENERATION_FILES.issubset(staged_names):
            raise ValueError("recording_generation_required_file_missing")
        if not staged_names.issubset(ALLOWED_GENERATION_FILES):
            raise ValueError("recording_generation_contains_unexpected_file")

        files: dict[str, dict[str, Any]] = {}
        for name in sorted(staged_names):
            path = staging / name
            try:
                info = path.lstat()
            except OSError as exc:
                raise ValueError("recording_generation_file_unreadable") from exc
            if path.is_symlink() or not path.is_file() or info.st_size <= 0:
                raise ValueError("recording_generation_file_invalid")
            if int(info.st_size) > self.max_bytes:
                raise ValueError("recording_generation_exceeds_cache_limit")
            files[name] = {
                "size": int(info.st_size),
                "mtime_ns": int(info.st_mtime_ns),
                "sha256": _sha256_file(path),
            }
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        generation_id = hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "complete": True,
            "generation_id": generation_id,
            "source_fingerprint": entry.source_fingerprint,
            "profile_fingerprint": entry.profile_fingerprint,
            "created_at_ns": time.time_ns(),
            "files": files,
        }
        _atomic_json(staging / "manifest.json", manifest)
        _fsync_directory(staging)
        if entry.directory.exists():
            raise ValueError("recording_generation_already_exists")
        os.replace(staging, entry.directory)
        _fsync_directory(entry.directory.parent)
        manifest_info = entry.manifest_path.stat()
        with self._verified_guard:
            self._verified.add((str(entry.directory), int(manifest_info.st_mtime_ns)))

    def _load(self, entry: CacheEntry, *, verify_hash: bool, touch: bool = True) -> CacheLoad:
        try:
            if entry.directory.is_symlink() or not entry.directory.is_dir():
                return CacheLoad("miss")
        except OSError:
            return CacheLoad("miss")
        manifest = _safe_json(entry.manifest_path, max_bytes=MAX_MANIFEST_BYTES)
        if (
            manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
            or manifest.get("cache_schema_version") != CACHE_SCHEMA_VERSION
            or manifest.get("complete") is not True
            or manifest.get("source_fingerprint") != entry.source_fingerprint
            or manifest.get("profile_fingerprint") != entry.profile_fingerprint
        ):
            return CacheLoad("corrupt")
        files = manifest.get("files")
        file_names = set(files) if isinstance(files, Mapping) else set()
        if (
            not isinstance(files, Mapping)
            or not REQUIRED_GENERATION_FILES.issubset(file_names)
            or not file_names.issubset(ALLOWED_GENERATION_FILES)
        ):
            return CacheLoad("corrupt")
        try:
            total_bytes = int(entry.manifest_path.stat().st_size)
        except OSError:
            return CacheLoad("corrupt")
        for name in sorted(file_names):
            expected = files.get(name)
            if not isinstance(expected, Mapping):
                return CacheLoad("corrupt")
            path = entry.directory / name
            try:
                info = path.lstat()
            except OSError:
                return CacheLoad("corrupt")
            if (
                path.is_symlink()
                or not path.is_file()
                or int(expected.get("size") or -1) != int(info.st_size)
                or int(expected.get("mtime_ns") or -1) != int(info.st_mtime_ns)
            ):
                return CacheLoad("corrupt")
            total_bytes += int(info.st_size)
        created_at_ns = int(manifest.get("created_at_ns") or 0)
        last_access_ns = self._last_access_ns(entry, created_at_ns)
        if self.ttl_s > 0 and last_access_ns > 0 and time.time_ns() - last_access_ns > int(self.ttl_s * 1e9):
            return CacheLoad("expired", total_bytes=total_bytes)
        try:
            manifest_mtime = int(entry.manifest_path.stat().st_mtime_ns)
        except OSError:
            return CacheLoad("corrupt")
        verification_key = (str(entry.directory), manifest_mtime)
        with self._verified_guard:
            verified = verification_key in self._verified
        if not verified and not verify_hash:
            return CacheLoad("unverified", total_bytes=total_bytes)
        if not verified:
            for name in sorted(file_names):
                expected = files[name]
                try:
                    actual = _sha256_file(entry.directory / name)
                except OSError:
                    return CacheLoad("corrupt")
                if actual != expected.get("sha256"):
                    return CacheLoad("corrupt")
            with self._verified_guard:
                self._verified.add(verification_key)
        summary = _safe_json(entry.summary_path, max_bytes=MAX_JSON_BYTES)
        if (
            summary.get("schema_version") != CACHE_SCHEMA_VERSION
            or summary.get("source_fingerprint") != entry.source_fingerprint
            or summary.get("profile_fingerprint") != entry.profile_fingerprint
            or not isinstance(summary.get("inspection"), Mapping)
        ):
            return CacheLoad("corrupt")
        if touch:
            self._touch_access(entry)
        return CacheLoad("ready", dict(summary["inspection"]), total_bytes)

    def load(self, entry: CacheEntry, *, verify_hash: bool = False) -> CacheLoad:
        return self._load(entry, verify_hash=verify_hash)

    def quarantine_locked(self, entry: CacheEntry, *, reason: str) -> Optional[Path]:
        if not entry.directory.exists():
            return None
        token = hashlib.sha256(f"{entry.directory}:{time.time_ns()}".encode("utf-8")).hexdigest()[:20]
        destination = self.quarantine_root / token
        try:
            _atomic_json(
                entry.directory / "quarantine.json",
                {"schema_version": 1, "reason": reason[:100], "quarantined_at_ns": time.time_ns()},
            )
            os.replace(entry.directory, destination)
            _fsync_directory(entry.directory.parent)
            return destination
        except OSError:
            return None

    def remove_complete_locked(self, entry: CacheEntry) -> bool:
        load = self._load(entry, verify_hash=False, touch=False)
        if load.status not in {"ready", "unverified", "expired"}:
            return False
        shutil.rmtree(entry.directory, ignore_errors=True)
        _fsync_directory(entry.directory.parent)
        return not entry.directory.exists()

    def remove_episode(self, episode: Path) -> bool:
        """Remove every replay-cache generation for one trusted source path."""
        episode_key = self._episode_key(episode)
        directory = self.entries_root / episode_key[:2] / episode_key
        with self.lock(f"episode:{episode_key}", timeout_s=None) as acquired:
            if not acquired:  # pragma: no cover - blocking lock always acquires.
                return False
            shutil.rmtree(directory, ignore_errors=True)
            try:
                directory.parent.rmdir()
            except OSError:
                pass
            return not directory.exists()

    @staticmethod
    def _presentation_fingerprint(path: Path) -> str:
        identities: list[tuple[str, int, int]] = []
        for candidate in (
            path,
            path / "metadata.json",
            path / "episode_result.json",
            path / "episode_incomplete.json",
            path / ".conversion_source.json",
            path / "dense_labels.jsonl",
        ):
            try:
                info = candidate.lstat()
            except OSError:
                continue
            if candidate != path and (candidate.is_symlink() or not candidate.is_file()):
                continue
            identities.append(
                ("." if candidate == path else candidate.name, int(info.st_size), int(info.st_mtime_ns))
            )
        encoded = json.dumps(identities, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def episode_index(
        self,
        root: Path,
        *,
        name_is_valid: Callable[[str], bool],
        metadata_loader: Callable[[Path], Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        root = Path(root).resolve(strict=True)
        root_key = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
        index_path = self.roots_root / root_key / "index.json"
        with self.lock(f"index:{root_key}", timeout_s=DEFAULT_LOCK_WAIT_S) as acquired:
            cached = _safe_json(index_path, max_bytes=MAX_JSON_BYTES) if acquired else {}
            cached_items = cached.get("episodes") if cached.get("schema_version") == CACHE_SCHEMA_VERSION else {}
            if not isinstance(cached_items, Mapping):
                cached_items = {}
            try:
                children = list(root.iterdir())
            except OSError:
                return []
            next_items: dict[str, dict[str, Any]] = {}
            for child in children:
                if not name_is_valid(child.name):
                    continue
                try:
                    if child.is_symlink() or not child.is_dir():
                        continue
                except OSError:
                    continue
                presentation = self._presentation_fingerprint(child)
                previous = cached_items.get(child.name)
                if (
                    isinstance(previous, Mapping)
                    and previous.get("presentation_fingerprint") == presentation
                    and isinstance(previous.get("item"), Mapping)
                ):
                    item = dict(previous["item"])
                else:
                    item = dict(metadata_loader(child))
                next_items[child.name] = {"presentation_fingerprint": presentation, "item": item}
            if acquired:
                payload = {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "root_key": root_key,
                    "updated_at_ns": time.time_ns(),
                    "episodes": next_items,
                }
                if cached_items != next_items or cached.get("root_key") != root_key:
                    _atomic_json(index_path, payload)
            return [dict(value["item"]) for value in next_items.values()]

    def cleanup(self, *, keep: Optional[CacheEntry] = None) -> None:
        """Bound all cache generations and auxiliary state without blocking builders."""
        keep_path = keep.directory.resolve(strict=False) if keep is not None else None
        with self.lock("cleanup", timeout_s=0.0) as cleanup_acquired:
            if not cleanup_acquired:
                return
            now = time.time()
            stale_work_s = min(self.ttl_s, STALE_WORK_TTL_S) if self.ttl_s > 0 else STALE_WORK_TTL_S
            entry_records: list[tuple[int, int, CacheEntry]] = []
            auxiliary_records: list[tuple[int, int, Path, str]] = []
            expired_access: list[Path] = []
            total = 0

            def tree_size(path: Path) -> int:
                size = 0
                try:
                    for child in path.rglob("*"):
                        if child.is_file() and not child.is_symlink():
                            size += int(child.stat().st_size)
                except OSError:
                    pass
                return size

            # A crashed worker can leave shadows or staging directories.  Never
            # touch young work and take the corresponding entry lock for staging.
            with self.lock("shadow-namespace", timeout_s=None) as shadow_acquired:
                if shadow_acquired:
                    try:
                        shadows = list(self.shadows_root.iterdir())
                    except OSError:
                        shadows = []
                    for shadow in shadows:
                        if shadow.is_symlink() or not shadow.is_dir():
                            continue
                        try:
                            modified = shadow.stat().st_mtime
                        except OSError:
                            continue
                        size = tree_size(shadow)
                        if now - modified > stale_work_s:
                            shutil.rmtree(shadow, ignore_errors=True)
                        else:
                            total += size
                    try:
                        self.shadows_root.rmdir()
                    except OSError:
                        pass
            for staging in self.entries_root.glob("*/*/*/.*.build.*"):
                if staging.is_symlink() or not staging.is_dir() or ".build." not in staging.name:
                    continue
                try:
                    modified = staging.stat().st_mtime
                except OSError:
                    continue
                size = tree_size(staging)
                profile = staging.name[1:].split(".build.", 1)[0]
                final_entry = CacheEntry(staging.parent / profile, staging.parent.name, profile)
                if now - modified > stale_work_s:
                    with self.entry_lock(final_entry, timeout_s=0.0) as acquired:
                        if acquired:
                            shutil.rmtree(staging, ignore_errors=True)
                else:
                    total += size

            # Quarantined generations, persisted job state, and access markers
            # are part of the same byte/TTL budget as ready generations.
            for quarantined in self.quarantine_root.iterdir():
                if quarantined.is_symlink() or not quarantined.is_dir():
                    continue
                try:
                    modified_ns = int(quarantined.stat().st_mtime_ns)
                except OSError:
                    continue
                size = tree_size(quarantined)
                if self.ttl_s > 0 and time.time_ns() - modified_ns > int(self.ttl_s * 1e9):
                    shutil.rmtree(quarantined, ignore_errors=True)
                else:
                    total += size
                    auxiliary_records.append((modified_ns, size, quarantined, "tree"))
            for root in (self.states_root, self.access_root):
                for item in root.iterdir():
                    if item.is_symlink() or not item.is_file():
                        continue
                    try:
                        info = item.stat()
                    except OSError:
                        continue
                    modified_ns = int(info.st_mtime_ns)
                    size = int(info.st_size)
                    if self.ttl_s > 0 and time.time_ns() - modified_ns > int(self.ttl_s * 1e9):
                        if root == self.access_root:
                            # Entry TTL evaluation below still needs this marker.
                            expired_access.append(item)
                        else:
                            item.unlink(missing_ok=True)
                    else:
                        total += size
                        auxiliary_records.append((modified_ns, size, item, "file"))

            for directory in self.entries_root.glob("*/*/*/*"):
                if directory.is_symlink() or not directory.is_dir() or ".build." in directory.name:
                    continue
                entry = CacheEntry(directory, directory.parent.name, directory.name)
                with self.entry_lock(entry, timeout_s=0.0) as acquired:
                    if not acquired:
                        continue
                    load = self._load(entry, verify_hash=False, touch=False)
                    if load.status not in {"ready", "unverified", "expired"}:
                        continue
                    manifest = _safe_json(entry.manifest_path, max_bytes=MAX_MANIFEST_BYTES)
                    created_at_ns = int(manifest.get("created_at_ns") or 0)
                    last_access_ns = self._last_access_ns(entry, created_at_ns)
                    if keep_path != directory.resolve(strict=False) and load.status == "expired":
                        self.remove_complete_locked(entry)
                        continue
                    total += load.total_bytes
                    entry_records.append((last_access_ns, load.total_bytes, entry))
            for item in expired_access:
                item.unlink(missing_ok=True)
            if total <= self.max_bytes:
                return

            # Quarantine has no serving value, so evict it before the LRU of
            # complete generations.  Persisted state/access files go last.
            for _modified, size, path, kind in sorted(
                (record for record in auxiliary_records if record[3] == "tree"),
                key=lambda item: item[0],
            ):
                shutil.rmtree(path, ignore_errors=True)
                total -= size
                if total <= self.max_bytes:
                    return
            for _accessed, size, entry in sorted(entry_records, key=lambda item: item[0]):
                if keep_path == entry.directory.resolve(strict=False):
                    continue
                with self.entry_lock(entry, timeout_s=0.0) as acquired:
                    if not acquired:
                        continue
                    if self.remove_complete_locked(entry):
                        total -= size
                if total <= self.max_bytes:
                    return
            for _modified, size, path, kind in sorted(
                (record for record in auxiliary_records if record[3] == "file"),
                key=lambda item: item[0],
            ):
                path.unlink(missing_ok=True)
                total -= size
                if total <= self.max_bytes:
                    return
