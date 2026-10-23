from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from incubator.control_panel.converter_profile import (
    ACTIVATION_SCHEMA_VERSION,
    ConverterProfile,
    ConverterProfilePreflight,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile(tmp_path: Path) -> ConverterProfile:
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True)
    python = runtime / "python"
    source = tmp_path / "converter.py"
    merger = tmp_path / "merger.py"
    wrapper = tmp_path / "wrapper.py"
    ffmpeg = runtime / "ffmpeg"
    ffprobe = runtime / "ffprobe"
    decoder = runtime / "decode-ffmpeg"
    for path in (python, source, merger, wrapper, ffmpeg, ffprobe, decoder):
        path.write_bytes(path.name.encode())
        path.chmod(0o755)
    return ConverterProfile(
        profile_id="test-profile",
        python=python,
        wrapper_script=wrapper,
        converter_source=source,
        converter_source_sha256=_hash(source),
        merger_source=merger,
        merger_source_sha256=_hash(merger),
        ffmpeg=ffmpeg,
        ffmpeg_sha256=_hash(ffmpeg),
        ffprobe=ffprobe,
        ffprobe_sha256=_hash(ffprobe),
        full_decode_ffmpeg=decoder,
        output_root=tmp_path / "output",
        activation_receipt=runtime / "activation.json",
        wrapper_script_sha256=_hash(wrapper),
        min_free_bytes=100,
    )


def _activation(profile: ConverterProfile) -> dict:
    return {
        "schema_version": ACTIVATION_SCHEMA_VERSION,
        "status": "passed",
        "profile_fingerprint": profile.fingerprint,
        "conversion": {
            "all_parquet_read": True,
            "all_videos_ffprobe": True,
            "all_videos_full_decode": True,
            "manifest_hashes_verified": True,
            "source_unchanged": True,
        },
        "merge": {
            "two_shards": True,
            "no_video_reencode": True,
            "all_parquet_read": True,
            "all_videos_ffprobe": True,
            "all_videos_full_decode": True,
            "manifest_hashes_verified": True,
        },
    }


class FakeRunner:
    def __init__(self, sdk_version: str = "0.2.18") -> None:
        self.sdk_version = sdk_version
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        command = list(command)
        self.commands.append(command)
        executable = Path(command[0]).name
        if executable == "python":
            stdout = json.dumps(
                {
                    "dm3_core": self.sdk_version,
                    "nmx_msg": "2.2.0",
                    "dm3_mcap": True,
                    "unified_reader_factory": True,
                    "decoded_messages": True,
                    "reader_config": True,
                }
            )
        elif executable == "ffprobe":
            stdout = json.dumps(
                {"streams": [{"codec_name": "av1", "pix_fmt": "yuv420p", "nb_frames": "1"}]}
            )
        else:
            stdout = ""
            if command[-1].endswith(".mp4"):
                Path(command[-1]).write_bytes(b"video")
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def test_activation_receipt_is_required_after_real_media_smoke(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    runner = FakeRunner()
    preflight = ConverterProfilePreflight(
        profile,
        process_runner=runner,
        disk_usage=lambda path: SimpleNamespace(free=10_000),
        status_cache_s=0,
        now=lambda: 1000.0,
    )
    available, blocker = preflight.available()
    assert available is False
    assert blocker == "awaiting_first_finalized_episode"
    assert preflight.status()["can_probe"] is True
    assert preflight.cache_path.is_file()
    profile.activation_receipt.write_text(json.dumps(_activation(profile)))
    assert preflight.available() == (True, "")
    assert sum(Path(command[0]).name == "ffmpeg" for command in runner.commands) == 2


def test_cached_media_smoke_is_bound_to_profile_fingerprint(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    profile.activation_receipt.write_text(json.dumps(_activation(profile)))
    runner = FakeRunner()
    preflight = ConverterProfilePreflight(
        profile,
        process_runner=runner,
        disk_usage=lambda path: SimpleNamespace(free=10_000),
        status_cache_s=0,
        now=lambda: 1000.0,
    )
    assert preflight.available() == (True, "")
    first_media_calls = len(
        [command for command in runner.commands if Path(command[0]).name != "python"]
    )
    assert preflight.available() == (True, "")
    assert len([command for command in runner.commands if Path(command[0]).name != "python"]) == first_media_calls


def test_hash_sdk_and_space_fail_closed(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    profile.activation_receipt.write_text(json.dumps(_activation(profile)))
    profile.converter_source.write_text("changed")
    mismatch = ConverterProfilePreflight(
        profile,
        process_runner=FakeRunner(),
        disk_usage=lambda path: SimpleNamespace(free=10_000),
        status_cache_s=0,
    )
    assert "checksum mismatch" in mismatch.available()[1]

    profile = _profile(tmp_path / "merger")
    profile.activation_receipt.write_text(json.dumps(_activation(profile)))
    profile.merger_source.write_text("changed")
    merger_mismatch = ConverterProfilePreflight(
        profile,
        process_runner=FakeRunner(),
        disk_usage=lambda path: SimpleNamespace(free=10_000),
        status_cache_s=0,
    )
    assert "merger source checksum mismatch" in merger_mismatch.available()[1]

    profile = _profile(tmp_path / "wrapper")
    profile.activation_receipt.write_text(json.dumps(_activation(profile)))
    profile.wrapper_script.write_text("changed")
    wrapper_mismatch = ConverterProfilePreflight(
        profile,
        process_runner=FakeRunner(),
        disk_usage=lambda path: SimpleNamespace(free=10_000),
        status_cache_s=0,
    )
    assert "accepted-episode wrapper checksum mismatch" in wrapper_mismatch.available()[1]

    profile = _profile(tmp_path / "sdk")
    profile.activation_receipt.write_text(json.dumps(_activation(profile)))
    sdk = ConverterProfilePreflight(
        profile,
        process_runner=FakeRunner("0.3.4"),
        disk_usage=lambda path: SimpleNamespace(free=10_000),
        status_cache_s=0,
    )
    assert "unexpected NEDF SDK" in sdk.available()[1]

    profile = _profile(tmp_path / "disk")
    profile.activation_receipt.write_text(json.dumps(_activation(profile)))
    disk = ConverterProfilePreflight(
        profile,
        process_runner=FakeRunner(),
        disk_usage=lambda path: SimpleNamespace(free=99),
        status_cache_s=0,
    )
    assert "requires 100" in disk.available()[1]


def test_activation_probe_failure_is_fingerprint_bound_and_disables_probe(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    runner = FakeRunner()
    preflight = ConverterProfilePreflight(
        profile,
        process_runner=runner,
        disk_usage=lambda path: SimpleNamespace(free=10_000),
        status_cache_s=0,
    )
    assert preflight.status()["can_probe"] is True
    preflight.record_probe_failure("a" * 64, "real conversion failed")
    status = preflight.status()
    assert status == {
        "available": False,
        "can_probe": False,
        "state": "activation_probe_failed",
        "blocker": "real conversion failed",
    }
