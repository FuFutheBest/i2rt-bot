import os
from pathlib import Path
import multiprocessing
import threading
import time

import pytest

from incubator.control_panel.recording_replay_cache import RecordingReplayCache


def _source(tmp_path: Path, name: str = "episode_0001") -> Path:
    episode = tmp_path / name
    data = episode / "data"
    data.mkdir(parents=True)
    (episode / "metadata.json").write_text('{"data_format":"NEDF3"}\n', encoding="utf-8")
    (data / "mcap_index.json").write_text('{"files":["data_0.mcap"]}\n', encoding="utf-8")
    (data / "data_0.mcap").write_bytes(b"mcap-source")
    return episode


def _shadow_process(cache_root, shadow_root, episode, start, inside):
    cache = RecordingReplayCache(Path(cache_root), shadow_root=Path(shadow_root))
    identity = cache.source_identity(Path(episode))
    start.wait(timeout=10)
    with cache.shadow_episode(Path(episode), identity) as shadow:
        assert shadow.is_dir()
        inside.wait(timeout=10)


def _root_layout_source(tmp_path: Path) -> Path:
    episode = tmp_path / "episode_root_layout"
    episode.mkdir(parents=True)
    (episode / "metadata.json").write_text('{"data_format":"NEDF3"}\n', encoding="utf-8")
    (episode / "mcap_index.json").write_text('[{"filename":"data_0.mcap"}]\n', encoding="utf-8")
    (episode / "data_0.mcap").write_bytes(b"root-layout-mcap")
    return episode


def test_source_identity_ignores_presentation_and_sdk_sidecars(tmp_path):
    episode = _source(tmp_path)
    cache = RecordingReplayCache(tmp_path / "cache")
    before = cache.source_identity(episode)
    (episode / "timestamps.json").write_text("{}\n", encoding="utf-8")
    (episode / "episode_result.json").write_text('{"episode_result":"success"}\n', encoding="utf-8")
    (episode / "dense_labels.jsonl").write_text("{}\n", encoding="utf-8")
    assert cache.source_identity(episode).fingerprint == before.fingerprint
    with (episode / "data" / "data_0.mcap").open("ab") as stream:
        stream.write(b"changed")
    assert cache.source_identity(episode).fingerprint != before.fingerprint


def test_source_identity_and_shadow_support_current_root_layout(tmp_path):
    episode = _root_layout_source(tmp_path)
    cache = RecordingReplayCache(tmp_path / "cache")
    assert not cache.shadows_root.exists()
    identity = cache.source_identity(episode)
    assert {item[0] for item in identity.artifacts} == {
        "metadata.json",
        "mcap_index.json",
        "data_0.mcap",
    }
    with cache.shadow_episode(episode, identity) as shadow:
        assert not shadow.name.startswith(".")
        assert "." not in shadow.name
        assert all("." not in part for part in shadow.parts if part not in {"/", ""})
        assert (shadow / "mcap_index.json").is_file()
        assert (shadow / "data_0.mcap").is_file()
        assert (shadow / "data" / "mcap_index.json").is_file()
        assert (shadow / "data" / "data_0.mcap").is_file()
        assert os.stat(shadow / "data_0.mcap").st_ino == os.stat(
            shadow / "data" / "data_0.mcap"
        ).st_ino
    assert not cache.shadows_root.exists()


def test_shadow_uses_external_tree_and_is_removed(tmp_path):
    episode = _source(tmp_path)
    cache = RecordingReplayCache(tmp_path / "cache")
    assert not cache.shadows_root.exists()
    identity = cache.source_identity(episode)
    source_mcap = episode / "data" / "data_0.mcap"
    with cache.shadow_episode(episode, identity) as shadow:
        assert not shadow.name.startswith(".")
        assert "." not in shadow.name
        assert all("." not in part for part in shadow.parts if part not in {"/", ""})
        shadow_mcap = shadow / "data" / "data_0.mcap"
        assert shadow.is_relative_to(cache.shadows_root)
        assert shadow_mcap.read_bytes() == source_mcap.read_bytes()
        if os.stat(shadow_mcap).st_dev == os.stat(source_mcap).st_dev:
            assert os.stat(shadow_mcap).st_ino == os.stat(source_mcap).st_ino
        (shadow / "timestamps.json").write_text("{}\n", encoding="utf-8")
    assert not shadow.exists()
    assert not cache.shadows_root.exists()
    assert not (episode / "timestamps.json").exists()


def test_shadow_root_with_dot_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="shadow_root_contains_dot"):
        RecordingReplayCache(
            tmp_path / "cache",
            shadow_root=tmp_path / ".unsafe-shadow-root",
        )


def test_shadow_namespace_create_and_rmdir_are_cross_process_serialized(tmp_path):
    first = _source(tmp_path / "first", "episode_first")
    second = _source(tmp_path / "second", "episode_second")
    cache_root = tmp_path / "cache"
    shadow_root = tmp_path / "shared_shadows"
    context = multiprocessing.get_context("spawn")
    start = context.Barrier(2)
    inside = context.Barrier(2)
    processes = [
        context.Process(
            target=_shadow_process,
            args=(cache_root, shadow_root, episode, start, inside),
        )
        for episode in (first, second)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    namespace = RecordingReplayCache(
        cache_root,
        shadow_root=shadow_root,
    ).shadows_root
    assert not namespace.exists()


def test_entry_lock_serializes_threads(tmp_path):
    cache = RecordingReplayCache(tmp_path / "cache")
    active = 0
    maximum = 0
    guard = threading.Lock()

    def worker():
        nonlocal active, maximum
        with cache.lock("same-entry"):
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with guard:
                active -= 1

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert maximum == 1


def test_cleanup_enforces_lru_size_cap(tmp_path):
    cache = RecordingReplayCache(tmp_path / "cache", max_bytes=900, access_touch_s=1)
    episode = _source(tmp_path)
    profile = cache.profile_fingerprint({"head": "camera_head"})
    first = cache.entry(episode, "a" * 64, profile)
    first_staging, first_temp = cache.prepare_staging(first)
    first_temp.write_bytes(b"1" * 700)
    cache.publish(first, inspection={"schema_version": 1}, staging=first_staging)

    second = cache.entry(episode, "b" * 64, profile)
    second_staging, second_temp = cache.prepare_staging(second)
    second_temp.write_bytes(b"2" * 700)
    cache.publish(second, inspection={"schema_version": 1}, staging=second_staging)
    cache.cleanup(keep=second)
    assert not first.directory.exists()
    assert second.timeline_path.is_file()


def test_expired_entry_is_not_returned(tmp_path):
    cache = RecordingReplayCache(tmp_path / "cache", ttl_s=1, access_touch_s=1)
    episode = _source(tmp_path)
    profile = cache.profile_fingerprint({})
    entry = cache.entry(episode, "c" * 64, profile)
    staging, temporary = cache.prepare_staging(entry)
    temporary.write_bytes(b"timeline")
    cache.publish(entry, inspection={"schema_version": 1}, staging=staging)
    assert cache.load(entry).status == "ready"
    old = time.time() - 10
    os.utime(cache._access_path(entry), (old, old))
    assert cache.load(entry).status == "expired"
    cache.cleanup()
    assert not entry.directory.exists()


def test_manifest_detects_corruption_on_cross_process_verification(tmp_path):
    cache = RecordingReplayCache(tmp_path / "cache")
    episode = _source(tmp_path)
    entry = cache.entry(episode, "d" * 64, cache.profile_fingerprint({}))
    staging, timeline = cache.prepare_staging(entry)
    timeline.write_bytes(b"original-timeline")
    cache.publish(entry, inspection={"schema_version": 1}, staging=staging)
    entry.timeline_path.write_bytes(b"corrupt-timeline!")
    reloaded = RecordingReplayCache(tmp_path / "cache")
    assert reloaded.load(entry, verify_hash=True).status == "corrupt"


def test_manifest_publishes_and_verifies_optional_cached_video(tmp_path):
    cache = RecordingReplayCache(tmp_path / "cache")
    episode = _source(tmp_path)
    entry = cache.entry(episode, "f" * 64, cache.profile_fingerprint({}))
    staging, timeline = cache.prepare_staging(entry)
    timeline.write_bytes(b"timeline")
    (staging / "video_left.mp4").write_bytes(b"h264-video")
    cache.publish(entry, inspection={"schema_version": 1}, staging=staging)

    assert entry.video_path("left").read_bytes() == b"h264-video"
    assert RecordingReplayCache(tmp_path / "cache").load(
        entry, verify_hash=True
    ).status == "ready"
    entry.video_path("left").write_bytes(b"corrupt!!!")
    assert RecordingReplayCache(tmp_path / "cache").load(
        entry, verify_hash=True
    ).status == "corrupt"


def test_cleanup_bounds_stale_auxiliary_cache_state(tmp_path):
    cache = RecordingReplayCache(tmp_path / "cache", ttl_s=1, max_bytes=1024)
    episode = _source(tmp_path)
    entry = cache.entry(episode, "e" * 64, cache.profile_fingerprint({}))
    staging, _timeline = cache.prepare_staging(entry)
    shadow = cache.shadows_root / ".stale-shadow"
    shadow.mkdir(parents=True)
    (shadow / "timestamps.json").write_bytes(b"x" * 100)
    quarantine = cache.quarantine_root / "stale-generation"
    quarantine.mkdir()
    (quarantine / "timeline.npz").write_bytes(b"x" * 100)
    cache.write_build_state(entry, "failed", retry_at_ns=1, error="failed")
    cache._touch_access(entry)
    old = time.time() - 10
    for path in (
        staging,
        shadow,
        quarantine,
        cache._state_path(entry),
        cache._access_path(entry),
    ):
        os.utime(path, (old, old))
    cache.cleanup()
    assert not staging.exists()
    assert not shadow.exists()
    assert not quarantine.exists()
    assert not cache._state_path(entry).exists()
    assert not cache._access_path(entry).exists()
