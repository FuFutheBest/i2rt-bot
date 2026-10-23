"""Dependency-free HTTP contracts for the SuperOffline browser service."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from scripts.analysis.superoffline import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
SUPER_OFFLINE_SCRIPT = REPO_ROOT / "scripts" / "analysis" / "superoffline.py"


def test_direct_superoffline_entrypoint_requires_runtime_profile() -> None:
    environment = os.environ.copy()
    environment.pop("SUPERINFERENCE_RUNTIME_ROLE", None)
    completed = subprocess.run(
        [sys.executable, str(SUPER_OFFLINE_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "runtime_profiles.py run --role superoffline" in completed.stderr


def _call(
    app: Any,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> tuple[str, dict[str, object] | str]:
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    response: dict[str, object] = {}
    parsed_path = urlsplit(path)

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        response["status"] = status
        response["headers"] = headers

    chunks = app(
        {
            "REQUEST_METHOD": method,
            "PATH_INFO": parsed_path.path,
            "QUERY_STRING": parsed_path.query,
            "CONTENT_LENGTH": str(len(body)),
            "wsgi.input": BytesIO(body),
        },
        start_response,
    )
    content = b"".join(chunks).decode("utf-8")
    if "application/json" in str(response["headers"]):
        return str(response["status"]), json.loads(content)
    return str(response["status"]), content


def _models() -> list[dict[str, str]]:
    return [
        {
            "adapter_id": "fastwam",
            "variant_id": "fastwam-memory",
            "checkpoint": "checkpoints/fastwam-memory",
        },
        {
            "adapter_id": "openpi",
            "config_name": "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
            "checkpoint": "checkpoints/openpi",
        },
    ]


def _job_until_finished(app: Any, job_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        status, payload = _call(app, "GET", f"/api/jobs/{job_id}")
        assert status == "200 OK"
        assert isinstance(payload, dict)
        if payload["status"] not in {"queued", "running", "cancelling"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("SuperOffline job did not finish")


def test_page_and_model_catalog_render_without_flask() -> None:
    app = create_app(runner=lambda _request: {"unused": True})

    page_status, page = _call(app, "GET", "/")
    catalog_status, catalog = _call(app, "GET", "/api/models")

    assert page_status == "200 OK"
    assert isinstance(page, str) and "SuperOffline" in page
    assert 'id="prompt"' in page
    assert 'id="service-health"' in page
    assert "Observation Input" in page
    assert "Checkpoints" in page
    assert "Run Inference" in page
    assert "loadObservations" in page
    assert "loadCheckpointCatalog" in page
    assert "Choose discovered checkpoint..." in page
    assert "Compare 2 to 4 checkpoints." in page
    assert 'id="addCheckpoint" type="button" class="quiet">' in page
    assert "renderTrajectory" in page
    assert "/api/observation/preview" in page
    assert 'id="smoothnessMetrics"' in page
    assert 'id="toggleVideos"' in page
    assert 'id="gpuMemory"' in page
    assert 'id="zeroOrigin"' in page
    assert 'id="plotLeft"' in page and 'id="plotRight"' in page
    assert 'data-value="stride"' in page
    assert 'id="jobSection"' in page
    assert 'id="validationWarnings"' in page
    assert "artifacts.input_preview" in page
    assert "artifacts.video_prediction" in page
    assert "Object.entries(artifacts)" not in page
    assert "payload.progress" in page
    assert catalog_status == "200 OK"
    assert isinstance(catalog, dict)
    assert {item["adapter_id"] for item in catalog["models"]} == {
        "diffusion",
        "fastwam",
        "lingbotva",
        "openpi",
    }
    diffusion = next(
        item for item in catalog["models"] if item["adapter_id"] == "diffusion"
    )
    assert diffusion["available"] is False
    assert "no reviewed TCP/UMI" in diffusion["availability_error"]
    assert "unavailableSelectedAdapter" in page


def test_workspace_discovery_endpoints_list_recordings_and_checkpoints(tmp_path: Path) -> None:
    observation = tmp_path / "recording"
    observation.mkdir()
    (observation / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "superinference.timestamped_observation",
                "stopped_at_ns": 123,
                "run_started_at_ns": 100,
                "frame_count": 1,
                "inference_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (observation / "inference.jsonl").write_text(
        json.dumps(
            {
                "kind": "inference",
                "timestamp_ns": 110,
                "file": "inference/000000.npz",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (observation / "inference").mkdir()
    (observation / "inference" / "000000.npz").write_bytes(b"placeholder")
    checkpoints = tmp_path / "checkpoints" / "model"
    checkpoints.mkdir(parents=True)
    (checkpoints / "config.yaml").write_text("framework: {}\n", encoding="utf-8")

    app = create_app(runner=lambda _request: {"unused": True})
    status, observations = _call(
        app,
        "GET",
        f"/api/observations?base_dir={observation.parent}",
    )
    assert status == "200 OK"
    assert isinstance(observations, list)
    assert observations and observations[0]["path"] == str(observation)

    status, samples = _call(
        app,
        "GET",
        f"/api/observations/samples?path={observation}",
    )
    assert status == "200 OK"
    assert samples == [
        {
            "file": "inference/000000.npz",
            "kind": "inference",
            "number": 0,
            "path": str(observation / "inference" / "000000.npz"),
            "timestamp_ns": 110,
        }
    ]

    status, discovered = _call(
        app,
        "GET",
        f"/api/checkpoints?fastwam_dir={checkpoints}",
    )
    assert status == "200 OK"
    assert isinstance(discovered, list)
    assert discovered[0]["path"] == str(checkpoints.resolve())

    app.service.shutdown()


def test_checkpoint_discovery_labels_experiments_instead_of_generic_steps(
    tmp_path: Path,
) -> None:
    fastwam_root = tmp_path / "fastwam"
    fastwam = fastwam_root / "gift" / "experiment_a" / "step_030000"
    fastwam.mkdir(parents=True)
    (fastwam / "config.yaml").write_text("framework: {}\n", encoding="utf-8")
    earlier_fastwam = fastwam.parent / "step_009000"
    earlier_fastwam.mkdir()
    (earlier_fastwam / "config.yaml").write_text(
        "framework: {}\n",
        encoding="utf-8",
    )

    openpi_root = tmp_path / "openpi"
    openpi = openpi_root / "gift_v1" / "99999"
    (openpi / "params").mkdir(parents=True)
    (openpi / "params" / "manifest").write_text("fixture", encoding="utf-8")
    asset = openpi / "assets" / "gift"
    asset.mkdir(parents=True)
    (asset / "norm_stats.json").write_text("{}", encoding="utf-8")

    app = create_app(runner=lambda _request: {"unused": True})
    status, discovered = _call(
        app,
        "GET",
        "/api/checkpoints?"
        f"fastwam_dir={fastwam_root}&openpi_dir={openpi_root}",
    )

    assert status == "200 OK"
    assert isinstance(discovered, list)
    fastwam_rows = [item for item in discovered if item["backend"] == "fastwam"]
    assert [item["display_name"] for item in fastwam_rows] == [
        "gift/experiment_a (step_009000)",
        "gift/experiment_a (step_030000)",
    ]
    assert [item["path"] for item in fastwam_rows] == [
        str(earlier_fastwam.resolve()),
        str(fastwam.resolve()),
    ]
    by_backend = {item["backend"]: item for item in discovered}
    assert by_backend["openpi"]["display_name"] == "gift_v1"
    assert by_backend["openpi"]["path"] == str(openpi.resolve())
    assert by_backend["openpi"]["pi_config_name"] == "default"
    assert by_backend["openpi"]["asset_id"] == "gift"
    app.service.shutdown()


def test_default_service_exposes_builtin_real_robot_scope_before_queueing() -> None:
    app = create_app()

    info_status, info = _call(app, "GET", "/api/service-info")
    health_status, health = _call(app, "GET", "/api/healthz")
    compare_status, compare = _call(
        app,
        "POST",
        "/api/compare",
        {"models": _models()},
    )

    assert info_status == "200 OK"
    assert health_status == "200 OK"
    assert isinstance(health, dict)
    assert health["status"] == "ok"
    assert health["service"] == "superoffline"
    assert health["scheduler"] == {
        "max_concurrent_jobs": 1,
        "queued_jobs": 0,
        "running_jobs": 0,
    }
    assert info == {
        "mode": "builtin-real-robot-offline",
        "requires_same_adapter": False,
        "supported_adapter_ids": ["fastwam", "lingbotva", "openpi"],
        "scheduler": {
            "max_concurrent_jobs": 1,
            "queued_jobs": 0,
            "running_jobs": 0,
        },
    }
    assert compare_status == "400 Bad Request"
    assert isinstance(compare, dict)
    assert "observation_path is required" in str(compare["error"])


def test_compare_api_rejects_an_unavailable_real_adapter() -> None:
    app = create_app(runner=lambda _request: {"unused": True})
    models = _models()
    models[1] = {
        "adapter_id": "diffusion",
        "checkpoint": "checkpoints/diffusion",
    }

    status, payload = _call(app, "POST", "/api/compare", {"models": models})

    assert status == "400 Bad Request"
    assert isinstance(payload, dict)
    assert "no reviewed TCP/UMI" in str(payload["error"])


def test_compare_api_requires_two_to_four_models() -> None:
    app = create_app(runner=lambda _request: {"unused": True})

    status, payload = _call(app, "POST", "/api/compare", {"models": _models()[:1]})

    assert status == "400 Bad Request"
    assert isinstance(payload, dict)
    assert "two to four" in str(payload["error"])


def test_validate_endpoint_checks_input_without_queueing() -> None:
    app = create_app(runner=lambda _request: {"unused": True})

    status, payload = _call(
        app,
        "POST",
        "/api/validate",
        {"observation_path": "logs/run", "models": _models()},
    )

    assert status == "200 OK"
    assert isinstance(payload, dict)
    assert payload["valid"] is True
    assert payload["warnings"] == []
    jobs_status, jobs = _call(app, "GET", "/api/jobs")
    assert jobs_status == "200 OK"
    assert jobs == {"jobs": []}


def test_compare_api_executes_an_injected_pair_runner() -> None:
    seen: list[object] = []

    def runner(request: object) -> dict[str, object]:
        seen.append(request)
        return {"artifacts": ["comparison.json"], "sample_count": 3}

    app = create_app(runner=runner)
    status, queued = _call(
        app,
        "POST",
        "/api/compare",
        {"observation_path": "logs/run", "models": _models()},
    )

    assert status == "202 Accepted"
    assert isinstance(queued, dict)
    finished = _job_until_finished(app, str(queued["job_id"]))
    assert finished["status"] == "complete"
    assert finished["result"] == {
        "artifacts": ["comparison.json"],
        "sample_count": 3,
    }
    assert len(seen) == 1


def test_job_events_and_non_blocking_warnings_are_exposed() -> None:
    class EventRunner:
        def validate_request(self, _request: object) -> None:
            return None

        def request_warnings(self, _request: object) -> list[dict[str, object]]:
            return [
                {
                    "code": "task_mismatch",
                    "severity": "warning",
                    "message": "Checkpoint task differs; inference will continue.",
                }
            ]

        def run(
            self,
            _request: object,
            *,
            event_callback: Any,
            cancel_event: threading.Event,
        ) -> dict[str, object]:
            assert not cancel_event.is_set()
            event_callback(
                {
                    "stage": "infer_action",
                    "progress": 63,
                    "latest_checkpoint": "FastWAM",
                    "latest_observation": "4",
                    "message": "Action inference started",
                }
            )
            return {"observation_count": 1}

    app = create_app(runner=EventRunner())
    status, validated = _call(
        app,
        "POST",
        "/api/validate",
        {"observation_path": "logs/run", "models": _models()},
    )
    assert status == "200 OK"
    assert isinstance(validated, dict)
    assert validated["valid"] is True
    assert validated["warnings"][0]["code"] == "task_mismatch"

    status, queued = _call(
        app,
        "POST",
        "/api/compare",
        {"observation_path": "logs/run", "models": _models()},
    )
    assert status == "202 Accepted"
    assert isinstance(queued, dict)
    finished = _job_until_finished(app, str(queued["job_id"]))
    assert finished["status"] == "complete"
    assert finished["stage"] == "complete"
    assert finished["progress"] == 100
    assert finished["latest_checkpoint"] == "FastWAM"
    assert finished["latest_observation"] == "4"
    assert any(
        item["stage"] == "infer_action" for item in finished["logs"]
    )
    assert finished["warnings"][0]["severity"] == "warning"
    app.service.shutdown()


def test_running_job_can_be_cancelled() -> None:
    started = threading.Event()

    def runner(
        _request: object,
        *,
        event_callback: Any,
        cancel_event: threading.Event,
    ) -> dict[str, object]:
        event_callback(
            {
                "stage": "load_checkpoint",
                "progress": 20,
                "message": "Loading checkpoint",
            }
        )
        started.set()
        assert cancel_event.wait(timeout=2.0)
        raise RuntimeError("cancelled")

    app = create_app(runner=runner)
    status, queued = _call(
        app,
        "POST",
        "/api/compare",
        {"observation_path": "logs/run", "models": _models()},
    )
    assert status == "202 Accepted"
    assert isinstance(queued, dict)
    assert started.wait(timeout=2.0)

    cancel_status, cancelling = _call(
        app,
        "POST",
        f"/api/jobs/{queued['job_id']}/cancel",
    )
    assert cancel_status == "200 OK"
    assert isinstance(cancelling, dict)
    assert cancelling["status"] == "cancelling"
    finished = _job_until_finished(app, str(queued["job_id"]))
    assert finished["status"] == "cancelled"
    assert finished["stage"] == "cancelled"
    assert any(item["stage"] == "cancelling" for item in finished["logs"])
    app.service.shutdown()


def test_trajectory_endpoint_reports_paths_and_chunk_smoothness(
    tmp_path: Path,
) -> None:
    models: list[dict[str, object]] = []
    for index, label in enumerate(("memory", "faster")):
        model_dir = tmp_path / f"model_{index}"
        model_dir.mkdir()
        action = np.zeros((5, 16), dtype=np.float64)
        action[:, 0] = np.arange(5, dtype=np.float64) * (index + 1)
        action[:, 8] = np.arange(5, dtype=np.float64) * (index + 2)
        np.savez_compressed(model_dir / "action.npz", canonical_action=action)
        (model_dir / "summary.json").write_text(
            json.dumps(
                {
                    "artifacts": {
                        "npz": "action.npz",
                        "canonical_action_key": "canonical_action",
                    }
                }
            ),
            encoding="utf-8",
        )
        models.append(
            {
                "model": {"label": label},
                "output_dir": model_dir.name,
            }
        )

    def runner(_request: object) -> dict[str, object]:
        return {"output_dir": str(tmp_path), "models": models}

    app = create_app(runner=runner)
    queued_status, queued = _call(
        app,
        "POST",
        "/api/compare",
        {"observation_path": "logs/run", "models": _models()},
    )
    assert queued_status == "202 Accepted"
    assert isinstance(queued, dict)
    finished = _job_until_finished(app, str(queued["job_id"]))
    assert finished["status"] == "complete"

    status, trajectory = _call(
        app,
        "GET",
        f"/api/jobs/{queued['job_id']}/trajectory",
    )
    assert status == "200 OK"
    assert isinstance(trajectory, dict)
    first = trajectory["trajectories"][0]
    assert first["steps"] == 5
    assert first["left"][-1] == [4.0, 0.0, 0.0]
    assert first["smoothness"]["left"] == {
        "path_length": 4.0,
        "velocity_rms": 1.0,
        "acceleration_rms": 0.0,
        "jerk_rms": 0.0,
        "roughness": 0.0,
    }

    status, smoothness = _call(
        app,
        "GET",
        f"/api/jobs/{queued['job_id']}/smoothness",
    )
    assert status == "200 OK"
    assert isinstance(smoothness, dict)
    assert smoothness["observation_count"] == 1
    assert [item["rank"] for item in smoothness["checkpoints"]] == [1, 2]
    assert smoothness["checkpoints"][0]["metrics"]["roughness"] == 0.0
    app.service.shutdown()


def test_compare_jobs_are_serialized() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    calls: list[object] = []

    def runner(request: object) -> dict[str, object]:
        calls.append(request)
        if len(calls) == 1:
            first_started.set()
            assert release_first.wait(timeout=2.0)
        return {"sample_count": len(calls)}

    app = create_app(runner=runner)
    first_status, first = _call(
        app,
        "POST",
        "/api/compare",
        {"observation_path": "logs/first", "models": _models()},
    )
    assert first_status == "202 Accepted"
    assert isinstance(first, dict)
    assert first_started.wait(timeout=2.0)

    second_status, second = _call(
        app,
        "POST",
        "/api/compare",
        {"observation_path": "logs/second", "models": _models()},
    )
    assert second_status == "202 Accepted"
    assert isinstance(second, dict)
    assert second["status"] == "queued"

    release_first.set()
    first_result = _job_until_finished(app, str(first["job_id"]))
    second_result = _job_until_finished(app, str(second["job_id"]))
    assert first_result["status"] == "complete"
    assert second_result["status"] == "complete"
    assert len(calls) == 2


def test_queued_job_can_be_cancelled_and_failed_job_retried() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    attempts = 0

    def runner(_request: object) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            first_started.set()
            assert release_first.wait(timeout=2.0)
            raise RuntimeError("test failure")
        return {"sample_count": 1}

    app = create_app(runner=runner)
    first_status, first = _call(
        app,
        "POST",
        "/api/compare",
        {"observation_path": "logs/first", "models": _models()},
    )
    assert first_status == "202 Accepted"
    assert isinstance(first, dict)
    assert first_started.wait(timeout=2.0)
    second_status, second = _call(
        app,
        "POST",
        "/api/compare",
        {"observation_path": "logs/second", "models": _models()},
    )
    assert second_status == "202 Accepted"
    assert isinstance(second, dict)
    cancelled_status, cancelled = _call(
        app, "POST", f"/api/jobs/{second['job_id']}/cancel"
    )
    assert cancelled_status == "200 OK"
    assert isinstance(cancelled, dict)
    assert cancelled["status"] == "cancelled"

    release_first.set()
    failed = _job_until_finished(app, str(first["job_id"]))
    assert failed["status"] == "error"
    retry_status, retried = _call(
        app, "POST", f"/api/jobs/{first['job_id']}/retry"
    )
    assert retry_status == "200 OK"
    assert isinstance(retried, dict)
    assert _job_until_finished(app, str(retried["job_id"]))["status"] == "complete"
