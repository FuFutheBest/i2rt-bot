"""Trusted runtime profile and cached preflight for NEDF3 conversion.

All executable paths and encoder arguments are host-owned constants.  The
browser may inspect the resulting status, but cannot choose a Python runtime,
media executable, output root, or encoder argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from utils.runtime_profiles import RuntimeProfileError, select_runtime_python


CONVERTER_SOURCE_SHA256 = "f9bd73384fae71938668d39af9b36d4883880b50cf7fe7b902f3c2c8162cbc11"
CONVERTER_WRAPPER_SHA256 = "7a8f4b904f2dede3a503df5ad70074c2364e41c84457f0fed9078e5ada1d9929"
MERGER_SOURCE_SHA256 = "966057a762817f596be9fd8a1d813631f1faf4bbe1a5b1f4c9b30dc7fcee5c28"
FFMPEG_SHA256 = "9a66a85caee6dc556b39fb000187849d2909ecf784a15bf51133901d76568dd4"
FFPROBE_SHA256 = "34b0c8c1133a4f36678612189d25d9d06012254c33d41b654e333dc74fd022c6"
PROFILE_SCHEMA_VERSION = 1
SMOKE_CACHE_SCHEMA_VERSION = 1
ACTIVATION_SCHEMA_VERSION = 1
DEFAULT_PROFILE_ID = "nedf3-finalized-lerobot-v2-av1-runtime-20260910"
DEFAULT_RUNTIME_ROOT = (
    Path.home() / ".local/share/superinference-control-panel"
)
DEFAULT_PYTHON = DEFAULT_RUNTIME_ROOT / "converter_py310/bin/python"
DEFAULT_FFMPEG = DEFAULT_RUNTIME_ROOT / "ffmpeg-av1/bin/ffmpeg"
DEFAULT_FFPROBE = DEFAULT_RUNTIME_ROOT / "ffmpeg-av1/bin/ffprobe"
DEFAULT_FULL_DECODE_FFMPEG = Path("/usr/bin/ffmpeg")
DEFAULT_OUTPUT_ROOT = DEFAULT_RUNTIME_ROOT / "conversion_jobs"
DEFAULT_ACTIVATION_RECEIPT = (
    DEFAULT_RUNTIME_ROOT / "converter_py310/activation-smoke.json"
)
DEFAULT_MIN_FREE_BYTES = 10 * 1024**3
MEDIA_SMOKE_MAX_AGE_S = 6 * 60 * 60

CONVERTER_RUNTIME_ROOT_ENV = "SUPERINFERENCE_CONVERTER_RUNTIME_ROOT"
CONVERTER_PYTHON_ENV = "SUPERINFERENCE_CONVERTER_PYTHON"
CONVERTER_FFMPEG_ENV = "SUPERINFERENCE_CONVERTER_FFMPEG"
CONVERTER_FFPROBE_ENV = "SUPERINFERENCE_CONVERTER_FFPROBE"
CONVERTER_FULL_DECODE_FFMPEG_ENV = (
    "SUPERINFERENCE_CONVERTER_FULL_DECODE_FFMPEG"
)
CONVERSION_OUTPUT_ROOT_ENV = "SUPERINFERENCE_CONVERSION_OUTPUT_ROOT"
CONVERTER_ACTIVATION_RECEIPT_ENV = (
    "SUPERINFERENCE_CONVERTER_ACTIVATION_RECEIPT"
)


class ConverterProfileError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _environment_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return fallback
    return Path(value.strip()).expanduser()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(_canonical_json(payload) + b"\n")
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
    info = path.lstat()
    if path.is_symlink() or not path.is_file() or info.st_size <= 0 or info.st_size > 1024 * 1024:
        raise ConverterProfileError(f"unsafe profile artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConverterProfileError(f"profile artifact is not an object: {path}")
    return value


@dataclass(frozen=True)
class ConverterProfile:
    profile_id: str
    python: Path
    wrapper_script: Path
    converter_source: Path
    converter_source_sha256: str
    merger_source: Path
    merger_source_sha256: str
    ffmpeg: Path
    ffmpeg_sha256: str
    ffprobe: Path
    ffprobe_sha256: str
    full_decode_ffmpeg: Path
    output_root: Path
    activation_receipt: Path
    wrapper_script_sha256: str = CONVERTER_WRAPPER_SHA256
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES
    dm3_core_version: str = "0.2.18"
    message_sdk_version: str = "2.2.0"
    video_encoder: str = "auto"
    gpu_ids: tuple[int, ...] = (0,)
    nvenc_preset: str = "p4"
    svt_preset: int = 12
    video_quality: int = 30
    video_gop: int = 2

    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "python": str(self.python),
            "wrapper_script": str(self.wrapper_script),
            "wrapper_script_sha256": self.wrapper_script_sha256,
            "converter_source": str(self.converter_source),
            "converter_source_sha256": self.converter_source_sha256,
            "merger_source": str(self.merger_source),
            "merger_source_sha256": self.merger_source_sha256,
            "ffmpeg": str(self.ffmpeg),
            "ffmpeg_sha256": self.ffmpeg_sha256,
            "ffprobe": str(self.ffprobe),
            "ffprobe_sha256": self.ffprobe_sha256,
            "full_decode_ffmpeg": str(self.full_decode_ffmpeg),
            "output_root": str(self.output_root),
            "dm3_core_version": self.dm3_core_version,
            "message_sdk_version": self.message_sdk_version,
            "video": {
                "encoder": self.video_encoder,
                "gpu_ids": list(self.gpu_ids),
                "nvenc_preset": self.nvenc_preset,
                "svt_preset": self.svt_preset,
                "quality": self.video_quality,
                "gop": self.video_gop,
            },
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.identity())).hexdigest()

    @property
    def activation_failure_receipt(self) -> Path:
        return self.activation_receipt.with_name("activation-failure.json")


def default_profile(repo_root: Path | None = None) -> ConverterProfile:
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    runtime_root = _environment_path(
        CONVERTER_RUNTIME_ROOT_ENV, DEFAULT_RUNTIME_ROOT
    )
    try:
        selection = select_runtime_python(
            "converter",
            environment_names=(CONVERTER_PYTHON_ENV,),
            config_name=os.environ.get("SUPERINFERENCE_CONFIG_NAME"),
            fallback=runtime_root / "converter_py310/bin/python",
            repo_root=root,
            require_verified_profile=True,
        )
    except RuntimeProfileError as exc:
        raise ConverterProfileError(f"converter Python profile is invalid: {exc}") from exc
    assert selection is not None
    python = selection.python
    ffmpeg = _environment_path(
        CONVERTER_FFMPEG_ENV, runtime_root / "ffmpeg-av1/bin/ffmpeg"
    )
    ffprobe = _environment_path(
        CONVERTER_FFPROBE_ENV, runtime_root / "ffmpeg-av1/bin/ffprobe"
    )
    activation_receipt = _environment_path(
        CONVERTER_ACTIVATION_RECEIPT_ENV,
        runtime_root / "converter_py310/activation-smoke.json",
    )
    return ConverterProfile(
        profile_id=DEFAULT_PROFILE_ID,
        python=python,
        wrapper_script=root / "incubator/control_panel/tools/convert_accepted_episode.py",
        converter_source=root / "incubator/control_panel/tools/nedf3_rollout_to_lerobot.py",
        converter_source_sha256=CONVERTER_SOURCE_SHA256,
        merger_source=root / "incubator/control_panel/tools/merge_lerobot_shards.py",
        merger_source_sha256=MERGER_SOURCE_SHA256,
        ffmpeg=ffmpeg,
        ffmpeg_sha256=FFMPEG_SHA256,
        ffprobe=ffprobe,
        ffprobe_sha256=FFPROBE_SHA256,
        full_decode_ffmpeg=_environment_path(
            CONVERTER_FULL_DECODE_FFMPEG_ENV,
            DEFAULT_FULL_DECODE_FFMPEG,
        ),
        output_root=_environment_path(
            CONVERSION_OUTPUT_ROOT_ENV, DEFAULT_OUTPUT_ROOT
        ),
        activation_receipt=activation_receipt,
        wrapper_script_sha256=CONVERTER_WRAPPER_SHA256,
    )


class ConverterProfilePreflight:
    """Fail-closed preflight with a fingerprint-bound media smoke cache."""

    def __init__(
        self,
        profile: ConverterProfile,
        *,
        cache_path: Path | None = None,
        process_runner: Callable[..., Any] = subprocess.run,
        disk_usage: Callable[[Path], Any] = shutil.disk_usage,
        now: Callable[[], float] = time.time,
        status_cache_s: float = 30.0,
        media_smoke_max_age_s: float = MEDIA_SMOKE_MAX_AGE_S,
    ) -> None:
        self.profile = profile
        self.cache_path = Path(
            cache_path or profile.activation_receipt.with_name("media-smoke-cache.json")
        )
        self._process_runner = process_runner
        self._disk_usage = disk_usage
        self._now = now
        self.status_cache_s = max(0.0, float(status_cache_s))
        self.media_smoke_max_age_s = max(0.0, float(media_smoke_max_age_s))
        self._lock = threading.Lock()
        self._last_check_at = 0.0
        self._last_status: dict[str, Any] | None = None

    def available(self) -> tuple[bool, str]:
        status = self.status()
        return bool(status["available"]), str(status.get("blocker") or "")

    def status(self) -> dict[str, Any]:
        now = self._now()
        with self._lock:
            if self._last_status is not None and now - self._last_check_at < self.status_cache_s:
                return dict(self._last_status)
            try:
                self._run_runtime_preflight(now)
                failure = self._activation_failure()
                if failure is not None:
                    status = {
                        "available": False,
                        "can_probe": False,
                        "state": "activation_probe_failed",
                        "blocker": str(failure.get("error") or "activation probe failed"),
                    }
                else:
                    try:
                        self._check_activation_receipt()
                        status = {
                            "available": True,
                            "can_probe": False,
                            "state": "ready",
                            "blocker": "",
                        }
                    except FileNotFoundError:
                        status = {
                            "available": False,
                            "can_probe": True,
                            "state": "awaiting_first_finalized_episode",
                            "blocker": "awaiting_first_finalized_episode",
                        }
            except Exception as exc:
                status = {
                    "available": False,
                    "can_probe": False,
                    "state": "blocked",
                    "blocker": f"converter profile preflight failed: {exc}",
                }
            self._last_check_at = now
            self._last_status = status
            return dict(status)

    def invalidate(self) -> None:
        with self._lock:
            self._last_check_at = 0.0
            self._last_status = None

    def record_probe_failure(self, job_id: str, error: str) -> None:
        _atomic_write_json(
            self.profile.activation_failure_receipt,
            {
                "schema_version": ACTIVATION_SCHEMA_VERSION,
                "status": "failed",
                "profile_fingerprint": self.profile.fingerprint,
                "job_id": str(job_id),
                "error": str(error)[:1200],
                "failed_at": _utc_now(),
            },
        )
        self.invalidate()

    def _activation_failure(self) -> dict[str, Any] | None:
        try:
            failure = _read_json(self.profile.activation_failure_receipt)
        except FileNotFoundError:
            return None
        if (
            failure.get("schema_version") != ACTIVATION_SCHEMA_VERSION
            or failure.get("status") != "failed"
            or failure.get("profile_fingerprint") != self.profile.fingerprint
        ):
            raise ConverterProfileError("activation failure receipt is invalid or stale")
        return failure

    def _require_executable(self, path: Path, label: str) -> None:
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ConverterProfileError(f"{label} is unavailable: {path}")

    def _require_hash(self, path: Path, expected: str, label: str) -> None:
        if not path.is_file() or path.is_symlink():
            raise ConverterProfileError(f"{label} is missing or unsafe: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ConverterProfileError(f"{label} checksum mismatch: {actual}")

    def _run_checked(self, command: Sequence[str], *, timeout: float = 30.0) -> Any:
        completed = self._process_runner(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout,
        )
        if int(completed.returncode) != 0:
            detail = (completed.stderr or completed.stdout or "command failed").strip()[-800:]
            raise ConverterProfileError(detail)
        return completed

    def _check_sdk(self) -> None:
        probe = (
            "import json; from importlib.metadata import version; "
            "from dm3_core.nedf import NEDFFactory,NEDFReaderBase,NEDFReaderConfig,NEDFType; "
            "from dm3_core.mcap import ReaderFactory,WriterFactory,McapFileType; "
            "assert callable(getattr(NEDFFactory,'get_reader',None)); "
            "assert callable(getattr(NEDFReaderBase,'iter_decoded_messages',None)); "
            "assert hasattr(NEDFType,'NEDF3'); "
            "assert callable(NEDFReaderConfig); "
            "assert ReaderFactory and WriterFactory and McapFileType; "
            "print(json.dumps({'dm3_core':version('dm3-core'),"
            "'nmx_msg':version('nmx_msg'),"
            "'dm3_mcap':True,"
            "'unified_reader_factory':True,"
            "'decoded_messages':True,'reader_config':True}))"
        )
        completed = self._run_checked([str(self.profile.python), "-c", probe])
        try:
            result = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise ConverterProfileError("SDK API probe returned invalid JSON") from exc
        if result != {
            "dm3_core": self.profile.dm3_core_version,
            "nmx_msg": self.profile.message_sdk_version,
            "dm3_mcap": True,
            "unified_reader_factory": True,
            "decoded_messages": True,
            "reader_config": True,
        }:
            raise ConverterProfileError(f"unexpected NEDF SDK/API result: {result!r}")

    def _cached_media_smoke_valid(self, now: float) -> bool:
        try:
            cache = _read_json(self.cache_path)
            return (
                cache.get("schema_version") == SMOKE_CACHE_SCHEMA_VERSION
                and cache.get("status") == "passed"
                and cache.get("profile_fingerprint") == self.profile.fingerprint
                and now - float(cache.get("completed_at_unix")) <= self.media_smoke_max_age_s
                and cache.get("checks")
                == {
                    "av1_nvenc_one_frame": True,
                    "libsvtav1_one_frame": True,
                    "ffprobe_metadata": True,
                    "full_decode": True,
                }
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError, ConverterProfileError):
            return False

    def _probe_encoded_video(self, path: Path) -> None:
        completed = self._run_checked(
            [
                str(self.profile.ffprobe),
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,pix_fmt,nb_frames",
                "-of", "json",
                str(path),
            ]
        )
        try:
            streams = json.loads(completed.stdout).get("streams", [])
        except json.JSONDecodeError as exc:
            raise ConverterProfileError("ffprobe returned invalid JSON") from exc
        if len(streams) != 1 or streams[0].get("codec_name") != "av1":
            raise ConverterProfileError(f"unexpected encoded stream metadata: {streams!r}")
        self._run_checked(
            [
                str(self.profile.full_decode_ffmpeg),
                "-hide_banner", "-loglevel", "error",
                "-i", str(path),
                "-f", "null", "-",
            ]
        )

    def _run_media_smoke(self, now: float) -> None:
        self.profile.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(
            prefix=".profile-smoke-", dir=self.profile.output_root
        ) as temporary:
            root = Path(temporary)
            outputs = []
            nvenc = root / "nvenc.mp4"
            self._run_checked(
                [
                    str(self.profile.ffmpeg),
                    "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=320x240:r=1",
                    "-frames:v", "1", "-an", "-c:v", "av1_nvenc",
                    "-preset", self.profile.nvenc_preset,
                    "-gpu", str(self.profile.gpu_ids[0]),
                    "-pix_fmt", "yuv420p", str(nvenc),
                ],
                timeout=60.0,
            )
            outputs.append(nvenc)
            cpu = root / "svt.mp4"
            self._run_checked(
                [
                    str(self.profile.ffmpeg),
                    "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=320x240:r=1",
                    "-frames:v", "1", "-an", "-c:v", "libsvtav1",
                    "-preset", str(self.profile.svt_preset),
                    "-crf", str(self.profile.video_quality),
                    "-g", str(self.profile.video_gop),
                    "-pix_fmt", "yuv420p", str(cpu),
                ],
                timeout=60.0,
            )
            outputs.append(cpu)
            for output in outputs:
                self._probe_encoded_video(output)
        _atomic_write_json(
            self.cache_path,
            {
                "schema_version": SMOKE_CACHE_SCHEMA_VERSION,
                "status": "passed",
                "profile_fingerprint": self.profile.fingerprint,
                "completed_at": _utc_now(),
                "completed_at_unix": now,
                "checks": {
                    "av1_nvenc_one_frame": True,
                    "libsvtav1_one_frame": True,
                    "ffprobe_metadata": True,
                    "full_decode": True,
                },
            },
        )

    def _check_activation_receipt(self) -> None:
        receipt = _read_json(self.profile.activation_receipt)
        required_conversion = {
            "all_parquet_read": True,
            "all_videos_ffprobe": True,
            "all_videos_full_decode": True,
            "manifest_hashes_verified": True,
            "source_unchanged": True,
        }
        required_merge = {
            "two_shards": True,
            "no_video_reencode": True,
            "all_parquet_read": True,
            "all_videos_ffprobe": True,
            "all_videos_full_decode": True,
            "manifest_hashes_verified": True,
        }
        if not (
            receipt.get("schema_version") == ACTIVATION_SCHEMA_VERSION
            and receipt.get("status") == "passed"
            and receipt.get("profile_fingerprint") == self.profile.fingerprint
            and receipt.get("conversion") == required_conversion
            and receipt.get("merge") == required_merge
        ):
            raise ConverterProfileError("end-to-end activation receipt is absent, stale, or incomplete")

    def _run_runtime_preflight(self, now: float) -> None:
        self._require_executable(self.profile.python, "converter Python")
        self._require_executable(self.profile.ffmpeg, "FFmpeg")
        self._require_executable(self.profile.ffprobe, "FFprobe")
        self._require_executable(self.profile.full_decode_ffmpeg, "full-decode FFmpeg")
        self._require_hash(
            self.profile.wrapper_script,
            self.profile.wrapper_script_sha256,
            "accepted-episode wrapper",
        )
        self._require_hash(
            self.profile.converter_source,
            self.profile.converter_source_sha256,
            "converter source",
        )
        self._require_hash(
            self.profile.merger_source,
            self.profile.merger_source_sha256,
            "merger source",
        )
        self._require_hash(self.profile.ffmpeg, self.profile.ffmpeg_sha256, "FFmpeg")
        self._require_hash(self.profile.ffprobe, self.profile.ffprobe_sha256, "FFprobe")
        self._check_sdk()
        self.profile.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        free = int(self._disk_usage(self.profile.output_root).free)
        if free < self.profile.min_free_bytes:
            raise ConverterProfileError(
                f"output root has {free} free bytes; requires {self.profile.min_free_bytes}"
            )
        if not self._cached_media_smoke_valid(now):
            self._run_media_smoke(now)


__all__ = [
    "ConverterProfile",
    "ConverterProfileError",
    "ConverterProfilePreflight",
    "default_profile",
]
