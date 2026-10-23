from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from incubator.control_panel.tools import export_accepted_lerobot_range as exporter


PROFILE_ID = "test-profile"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _accepted_episode(
    root: Path,
    name: str,
    *,
    timestamp: str,
    fingerprint: str,
    episode_index: int,
) -> Path:
    episode = root / name
    episode.mkdir(parents=True)
    _write_json(
        episode / "episode_result.json",
        {
            "episode_result": "success",
            "timestamp": timestamp,
            "episode_id": f"episode-id-{episode_index}",
            "episode_index": episode_index,
        },
    )
    _write_json(
        episode / ".conversion_source.json",
        {
            "schema_version": 1,
            "result": "success",
            "path": str(episode.resolve()),
            "episode_id": f"episode-id-{episode_index}",
            "episode_index": episode_index,
            "fingerprint": fingerprint,
            "dense_labels_sha256": "d" * 64,
            "finalized_at": timestamp,
        },
    )
    return episode


def _ready_job(conversion_root: Path, episode: Path, fingerprint: str) -> str:
    job_id = exporter.compute_job_id(fingerprint, PROFILE_ID)
    identity = {
        "source_fingerprint": fingerprint,
        "profile_id": PROFILE_ID,
    }
    _write_json(
        conversion_root / "jobs" / f"{job_id}.json",
        {
            "schema_version": 1,
            "job_id": job_id,
            "identity": identity,
            "state": "ready",
            "source": {
                "path": str(episode.resolve()),
                "fingerprint": fingerprint,
            },
        },
    )
    _write_json(
        conversion_root / "published" / job_id / "meta" / "shard_manifest.json",
        {
            "schema_version": 1,
            "job_id": job_id,
            "profile_id": PROFILE_ID,
            "source_fingerprint": fingerprint,
        },
    )
    return job_id


def test_job_id_uses_conversion_queue_canonical_identity() -> None:
    fingerprint = "a" * 64
    expected = hashlib.sha256(
        json.dumps(
            {
                "source_fingerprint": fingerprint,
                "profile_id": PROFILE_ID,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    assert exporter.compute_job_id(fingerprint, PROFILE_ID) == expected


def test_date_selection_keeps_only_whole_accepted_episodes(tmp_path: Path) -> None:
    recording_root = tmp_path / "rollouts"
    day = recording_root / "20260715"
    accepted = _accepted_episode(
        day,
        "episode_0000",
        timestamp="2026-07-15T09:30:00+08:00",
        fingerprint="a" * 64,
        episode_index=0,
    )
    _write_json(
        day / "episode_0001" / "episode_result.json",
        {"episode_result": "failure", "timestamp": "2026-07-15T10:00:00+08:00"},
    )
    _accepted_episode(
        recording_root / "20260714",
        "episode_0002",
        timestamp="2026-07-14T23:59:59+08:00",
        fingerprint="b" * 64,
        episode_index=2,
    )
    start, end, local_timezone = exporter.resolve_time_range(
        date_text="2026-07-15",
        start_text=None,
        end_text=None,
        timezone_name="Asia/Shanghai",
    )
    selected = exporter.discover_accepted_episodes(
        recording_root,
        start=start,
        end=end,
        local_timezone=local_timezone,
    )
    assert [item.path for item in selected] == [accepted.resolve()]


def test_explicit_selection_resolves_public_ids_and_sorts_by_timestamp(
    tmp_path: Path,
) -> None:
    recording_root = tmp_path / "rollouts"
    later = _accepted_episode(
        recording_root / "20260715",
        "episode_0002",
        timestamp="2026-07-15T11:00:00+08:00",
        fingerprint="b" * 64,
        episode_index=2,
    )
    earlier = _accepted_episode(
        recording_root,
        "episode_0001",
        timestamp="2026-07-15T10:00:00+08:00",
        fingerprint="a" * 64,
        episode_index=1,
    )
    selected = exporter.discover_selected_accepted_episodes(
        recording_root,
        ["20260715:episode_0002", "episode_0001"],
        local_timezone=ZoneInfo("Asia/Shanghai"),
    )
    assert [item.path for item in selected] == [earlier.resolve(), later.resolve()]


def test_new_episode_is_not_exportable_until_accept_decision_is_applied(
    tmp_path: Path,
) -> None:
    recording_root = tmp_path / "rollouts"
    episode = _accepted_episode(
        recording_root,
        "episode_0001",
        timestamp="2026-07-15T10:00:00+08:00",
        fingerprint="a" * 64,
        episode_index=1,
    )
    marker = json.loads((episode / ".conversion_source.json").read_text())
    marker["decision_required"] = True
    _write_json(episode / ".conversion_source.json", marker)
    with pytest.raises(exporter.ExportError, match="not accepted"):
        exporter.discover_selected_accepted_episodes(
            recording_root,
            ["episode_0001"],
            local_timezone=ZoneInfo("Asia/Shanghai"),
        )

    _write_json(
        episode / ".recording_decision.json",
        {"status": "accepted", "action": "episode_accept"},
    )
    selected = exporter.discover_selected_accepted_episodes(
        recording_root,
        ["episode_0001"],
        local_timezone=ZoneInfo("Asia/Shanghai"),
    )
    assert [item.path for item in selected] == [episode.resolve()]


@pytest.mark.parametrize(
    "episode_ids, message",
    [
        (["../episode_0001"], "Invalid selected episode id"),
        (["20260715:../episode_0001"], "Invalid selected episode id"),
        (["episode_0001", "episode_0001"], "must be unique"),
    ],
)
def test_explicit_selection_rejects_unsafe_or_duplicate_ids(
    tmp_path: Path,
    episode_ids: list[str],
    message: str,
) -> None:
    recording_root = tmp_path / "rollouts"
    _accepted_episode(
        recording_root,
        "episode_0001",
        timestamp="2026-07-15T10:00:00+08:00",
        fingerprint="a" * 64,
        episode_index=1,
    )
    with pytest.raises(exporter.ExportError, match=message):
        exporter.discover_selected_accepted_episodes(
            recording_root,
            episode_ids,
            local_timezone=ZoneInfo("Asia/Shanghai"),
        )


def test_explicit_selection_rejects_nonaccepted_episode(tmp_path: Path) -> None:
    recording_root = tmp_path / "rollouts"
    episode = recording_root / "episode_0001"
    _write_json(
        episode / "episode_result.json",
        {
            "episode_result": "failure",
            "timestamp": "2026-07-15T10:00:00+08:00",
        },
    )
    with pytest.raises(exporter.ExportError, match="not accepted"):
        exporter.discover_selected_accepted_episodes(
            recording_root,
            ["episode_0001"],
            local_timezone=ZoneInfo("Asia/Shanghai"),
        )


def test_inspection_reports_missing_job_without_dropping_episode(
    tmp_path: Path,
) -> None:
    recording_root = tmp_path / "rollouts"
    episode = _accepted_episode(
        recording_root,
        "episode_0000",
        timestamp="2026-07-15T09:30:00+08:00",
        fingerprint="a" * 64,
        episode_index=0,
    )
    candidate = exporter.AcceptedEpisode(
        path=episode.resolve(),
        timestamp=datetime(2026, 7, 15, 9, 30, tzinfo=ZoneInfo("Asia/Shanghai")),
        timestamp_source="episode_result.timestamp",
        episode_id="episode-id-0",
        episode_index=0,
    )
    result = exporter.inspect_conversion(
        candidate,
        conversion_root=tmp_path / "conversion_jobs",
        profile_id=PROFILE_ID,
    )
    assert result["state"] == "job_missing"
    assert result["path"] == str(episode.resolve())


def test_selection_fails_closed_on_unreadable_episode_result(tmp_path: Path) -> None:
    recording_root = tmp_path / "rollouts"
    result_path = recording_root / "20260715" / "episode_0000" / "episode_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text("not-json", encoding="utf-8")
    start, end, local_timezone = exporter.resolve_time_range(
        date_text="2026-07-15",
        start_text=None,
        end_text=None,
        timezone_name="Asia/Shanghai",
    )
    with pytest.raises(exporter.ExportError, match="Cannot safely classify"):
        exporter.discover_accepted_episodes(
            recording_root,
            start=start,
            end=end,
            local_timezone=local_timezone,
        )


def test_export_preserves_time_order_and_writes_selection_manifest(
    tmp_path: Path,
) -> None:
    recording_root = tmp_path / "rollouts"
    conversion_root = tmp_path / "conversion_jobs"
    later = _accepted_episode(
        recording_root / "20260715",
        "episode_0001",
        timestamp="2026-07-15T11:00:00+08:00",
        fingerprint="b" * 64,
        episode_index=1,
    )
    earlier = _accepted_episode(
        recording_root / "20260715",
        "episode_0000",
        timestamp="2026-07-15T10:00:00+08:00",
        fingerprint="a" * 64,
        episode_index=0,
    )
    first_job = _ready_job(conversion_root, earlier, "a" * 64)
    second_job = _ready_job(conversion_root, later, "b" * 64)
    start, end, local_timezone = exporter.resolve_time_range(
        date_text="2026-07-15",
        start_text=None,
        end_text=None,
        timezone_name="Asia/Shanghai",
    )
    episodes = exporter.discover_accepted_episodes(
        recording_root,
        start=start,
        end=end,
        local_timezone=local_timezone,
    )
    conversions = exporter.resolve_conversions(
        episodes,
        conversion_root=conversion_root,
        profile_id=PROFILE_ID,
    )
    captured: dict[str, object] = {}

    def fake_merge(shards, output_dir, dataset_name, *, link_mode):
        captured["shards"] = list(shards)
        captured["link_mode"] = link_mode
        root = Path(output_dir) / dataset_name
        (root / "meta").mkdir(parents=True)
        return root

    output = exporter.export_accepted_range(
        episodes,
        conversions,
        recording_root=recording_root,
        conversion_root=conversion_root,
        output_dir=tmp_path / "exports",
        dataset_name="accepted-20260715",
        start=start,
        end=end,
        timezone_name="Asia/Shanghai",
        profile_id=PROFILE_ID,
        link_mode="auto",
        merge_fn=fake_merge,
    )
    assert [path.name for path in captured["shards"]] == [first_job, second_job]
    assert captured["link_mode"] == "auto"
    manifest = json.loads((output / "meta" / "export_selection.json").read_text())
    assert manifest["episode_count"] == 2
    assert [item["job_id"] for item in manifest["episodes"]] == [
        first_job,
        second_job,
    ]
    assert manifest["selection"]["end_exclusive"] == end.isoformat()


def test_export_fails_when_any_selected_episode_is_not_ready(tmp_path: Path) -> None:
    episode = exporter.AcceptedEpisode(
        path=tmp_path / "episode_0000",
        timestamp=datetime(2026, 7, 15, tzinfo=ZoneInfo("Asia/Shanghai")),
        timestamp_source="episode_result.timestamp",
        episode_id="episode-id",
        episode_index=0,
    )
    with pytest.raises(exporter.ExportError, match="job_missing=1"):
        exporter.export_accepted_range(
            [episode],
            [{"state": "job_missing"}],
            recording_root=tmp_path,
            conversion_root=tmp_path / "conversion_jobs",
            output_dir=tmp_path / "exports",
            dataset_name="accepted",
            start=episode.timestamp,
            end=episode.timestamp.replace(hour=1),
            timezone_name="Asia/Shanghai",
            profile_id=PROFILE_ID,
            link_mode="auto",
            merge_fn=lambda *args, **kwargs: tmp_path / "unused",
        )
