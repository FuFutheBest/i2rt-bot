from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from incubator.control_panel import server
from incubator.control_panel.recording_exports import (
    RecordingExportError,
    RecordingExportManager,
)


PROFILE_ID = "test-profile"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _accepted_episode(root: Path, public_name: str = "episode_0001") -> Path:
    episode = root / public_name
    episode.mkdir(parents=True)
    fingerprint = "a" * 64
    timestamp = "2026-07-26T10:00:00+08:00"
    _write_json(
        episode / "episode_result.json",
        {
            "episode_result": "success",
            "episode_id": "episode-uuid-1",
            "episode_index": 1,
            "timestamp": timestamp,
        },
    )
    _write_json(
        episode / ".conversion_source.json",
        {
            "schema_version": 1,
            "result": "success",
            "path": str(episode.resolve()),
            "episode_id": "episode-uuid-1",
            "episode_index": 1,
            "fingerprint": fingerprint,
            "finalized_at": timestamp,
        },
    )
    return episode


def _conversion_manager(conversion_root: Path, python: Path) -> SimpleNamespace:
    return SimpleNamespace(
        queue_root=conversion_root,
        profile_id=PROFILE_ID,
        converter_python=python,
    )


def _publish_ready_conversion(conversion_root: Path, episode: Path) -> None:
    fingerprint = "a" * 64
    identity = {
        "source_fingerprint": fingerprint,
        "profile_id": PROFILE_ID,
    }
    job_id = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    _write_json(
        conversion_root / "jobs" / f"{job_id}.json",
        {
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
            "job_id": job_id,
            "profile_id": PROFILE_ID,
            "source_fingerprint": fingerprint,
        },
    )


def _fake_export_runner(command: list[str], **_kwargs: object) -> SimpleNamespace:
    output_root = Path(command[command.index("--output-dir") + 1])
    dataset_name = command[command.index("--dataset-name") + 1]
    dataset = output_root / dataset_name
    _write_json(dataset / "meta" / "info.json", {"total_episodes": 1})
    (dataset / "meta" / "episodes.jsonl").write_text(
        '{"episode_index":0}\n', encoding="utf-8"
    )
    _write_json(dataset / "meta" / "export_selection.json", {"episode_count": 1})
    _write_json(
        dataset / "meta" / "merge_manifest.json",
        {"video_materialization": {"hardlink": 3, "reflink": 0, "copy": 0}},
    )
    _write_json(
        dataset / "meta" / "validation_report.json",
        {"status": "passed", "episodes": 1, "videos": 3},
    )
    return SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr="")


def test_export_waits_for_idle_then_publishes_validated_dataset(
    tmp_path: Path,
) -> None:
    recording_root = tmp_path / "rollouts"
    episode = _accepted_episode(recording_root / "20260726")
    conversion_root = tmp_path / "conversion_jobs"
    _publish_ready_conversion(conversion_root, episode)
    busy = True
    manager = RecordingExportManager(
        lambda: recording_root,
        _conversion_manager(conversion_root, Path(__file__)),
        queue_root=tmp_path / "export_queue",
        output_root=tmp_path / "exports",
        export_python=Path(__file__),
        process_runner=_fake_export_runner,
        busy=lambda: busy,
        auto_start=False,
    )
    created = manager.enqueue(
        ["20260726:episode_0001"], dataset_name="selected-test"
    )
    assert created["episode_count"] == 1
    assert manager._eligible_job() is None
    waiting = manager.state()["jobs"][0]
    assert waiting["state"] == "waiting"
    assert waiting["phase"] == "waiting_idle"
    assert waiting["conversion_ready_count"] == 1

    busy = False
    selected = manager._eligible_job()
    assert selected is not None
    manager._execute(*selected)
    completed = manager.state()["jobs"][0]
    assert completed["state"] == "ready"
    assert completed["validation"]["status"] == "passed"
    assert completed["video_materialization"] == {
        "hardlink": 3,
        "reflink": 0,
        "copy": 0,
    }
    assert Path(completed["dataset_root"]).is_dir()


def test_export_does_not_merge_when_conversion_is_missing(tmp_path: Path) -> None:
    recording_root = tmp_path / "rollouts"
    _accepted_episode(recording_root)
    calls = 0

    def runner(*_args: object, **_kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    manager = RecordingExportManager(
        lambda: recording_root,
        _conversion_manager(tmp_path / "conversion_jobs", Path(__file__)),
        queue_root=tmp_path / "export_queue",
        output_root=tmp_path / "exports",
        export_python=Path(__file__),
        process_runner=runner,
        auto_start=False,
    )
    manager.enqueue(["episode_0001"], dataset_name="selected-test")
    assert manager._eligible_job() is None
    job = manager.state()["jobs"][0]
    assert job["state"] == "error"
    assert job["phase"] == "conversion_blocked"
    assert "job_missing=1" in job["error"]
    assert calls == 0


def test_export_rejects_nonaccepted_selection_and_existing_dataset(
    tmp_path: Path,
) -> None:
    recording_root = tmp_path / "rollouts"
    episode = recording_root / "episode_0001"
    _write_json(
        episode / "episode_result.json",
        {"episode_result": "failure", "timestamp": "2026-07-26T10:00:00+08:00"},
    )
    output_root = tmp_path / "exports"
    manager = RecordingExportManager(
        lambda: recording_root,
        _conversion_manager(tmp_path / "conversion_jobs", Path(__file__)),
        queue_root=tmp_path / "export_queue",
        output_root=output_root,
        export_python=Path(__file__),
        auto_start=False,
    )
    with pytest.raises(RecordingExportError, match="not accepted"):
        manager.enqueue(["episode_0001"], dataset_name="selected-test")

    episode_result = episode / "episode_result.json"
    payload = json.loads(episode_result.read_text(encoding="utf-8"))
    payload["episode_result"] = "success"
    payload["episode_id"] = "episode-uuid-1"
    payload["episode_index"] = 1
    _write_json(episode_result, payload)
    _write_json(
        episode / ".conversion_source.json",
        {
            "schema_version": 1,
            "result": "success",
            "path": str(episode.resolve()),
            "episode_id": "episode-uuid-1",
            "episode_index": 1,
            "fingerprint": "a" * 64,
        },
    )
    (output_root / "selected-test").mkdir(parents=True)
    with pytest.raises(RecordingExportError, match="already exists"):
        manager.enqueue(["episode_0001"], dataset_name="selected-test")


class _ExportApiStub:
    def __init__(self) -> None:
        self.request: tuple[list[str], object] | None = None

    def state(self, *, limit: int = 20) -> dict[str, object]:
        return {"available": True, "limit": limit, "jobs": []}

    def enqueue(self, episode_ids: list[str], *, dataset_name: object = None) -> dict[str, object]:
        self.request = (episode_ids, dataset_name)
        return {"job_id": "a" * 32, "state": "queued"}


def test_recording_export_api_is_async_and_accepts_ids_not_paths(tmp_path: Path) -> None:
    exports = _ExportApiStub()
    app = server.create_app(
        config_dir=tmp_path / "configs",
        recording_exports=exports,  # type: ignore[arg-type]
    )
    client = app.test_client()
    state = client.get("/api/recording-exports?limit=7")
    assert state.status_code == 200
    assert state.get_json()["limit"] == 7
    response = client.post(
        "/api/recording-exports",
        json={
            "episode_ids": ["20260726:episode_0001"],
            "dataset_name": "selected-test",
        },
    )
    assert response.status_code == 202
    assert exports.request == (["20260726:episode_0001"], "selected-test")
    invalid = client.post(
        "/api/recording-exports",
        json={"episode_ids": ["episode_0001"], "output_path": "/tmp/escape"},
    )
    assert invalid.status_code == 422
