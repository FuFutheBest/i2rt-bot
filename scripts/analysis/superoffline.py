#!/usr/bin/env python3
"""Serve the strict two-to-four-model SuperOffline comparison UI.

The service owns request validation and job lifecycle only. It deliberately
does not import policy classes or guess an interpreter. By default it runs any
registered compatible real-robot pair through their existing read-only evaluators. An
optional ``module:object`` executor may replace that integration. The service
always applies common trusted-input preflight first; the object then receives two
verified runtime contexts for its read-only model evaluation.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote
from uuid import uuid4
from wsgiref.simple_server import make_server

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if (
    __name__ == "__main__"
    and os.environ.get("SUPERINFERENCE_RUNTIME_ROLE") != "superoffline"
):
    print(
        "SuperOffline must be started through runtime_profiles.py run --role "
        "superoffline; register and bind a verified superoffline profile first.",
        file=sys.stderr,
    )
    raise SystemExit(2)

from scripts.analysis.superoffline_fastwam_executor import (  # noqa: E402
    RealRobotOfflineComparisonExecutor,
)
from utils import gpu_safety  # noqa: E402
from utils.openpi_checkpoint import (  # noqa: E402
    OpenPiCheckpointContractError,
    discover_openpi_norm_stats_assets,
)
from utils.superoffline_registry import (  # noqa: E402
    SUPER_OFFLINE_ADAPTERS,
    ComparisonExecutor,
    RuntimeProfileComparisonRunner,
    SuperOfflineAdapterRegistry,
    SuperOfflineComparisonRequest,
    SuperOfflineError,
)

_MAX_REQUEST_BYTES = 1_000_000


class SuperOfflineHttpError(SuperOfflineError):
    """A malformed browser request that should return HTTP 400."""


class SuperOfflineCapacityError(SuperOfflineError):
    """The bounded local comparison queue cannot accept another job."""


class SuperOfflineConflictError(SuperOfflineError):
    """A requested job transition is incompatible with its current state."""


@dataclass
class _ComparisonJob:
    job_id: str
    request: SuperOfflineComparisonRequest
    status: str
    submitted_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: Mapping[str, Any] | None = None
    error: str | None = None
    stage: str = "queued"
    progress: int = 0
    latest_checkpoint: str | None = None
    latest_observation: str | None = None
    heartbeat_at: str | None = None
    logs: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "request": self.request.as_dict(),
            "result": dict(self.result or {}),
            "error": self.error,
            "stage": self.stage,
            "progress": self.progress,
            "latest_checkpoint": self.latest_checkpoint,
            "latest_observation": self.latest_observation,
            "heartbeat_at": self.heartbeat_at,
            "logs": [dict(item) for item in self.logs],
            "warnings": [dict(item) for item in self.warnings],
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_runner(
    runner: Any,
    request: SuperOfflineComparisonRequest,
    *,
    event_callback: Callable[[Mapping[str, Any]], None],
    cancel_event: threading.Event,
) -> Mapping[str, Any]:
    run = getattr(runner, "run", None)
    target = run if callable(run) else runner
    kwargs: dict[str, Any] = {}
    try:
        parameters = inspect.signature(target).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    parameter_names = {parameter.name for parameter in parameters}
    accepts_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if accepts_keywords or "event_callback" in parameter_names:
        kwargs["event_callback"] = event_callback
    if accepts_keywords or "cancel_event" in parameter_names:
        kwargs["cancel_event"] = cancel_event
    result = target(request, **kwargs)
    if not isinstance(result, Mapping):
        raise TypeError("SuperOffline runner must return an object")
    return dict(result)


def _trajectory_smoothness(points: Any) -> dict[str, float]:
    """Return scale-transparent finite-difference metrics for one TCP chunk."""
    import numpy as np

    values = np.asarray(points, dtype=np.float64)

    def rms_norm(samples: Any) -> float:
        array = np.asarray(samples, dtype=np.float64)
        if not array.size:
            return 0.0
        return float(np.sqrt(np.mean(np.sum(array * array, axis=1))))

    velocity = np.diff(values, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)
    path_length = (
        float(np.linalg.norm(velocity, axis=1).sum()) if velocity.size else 0.0
    )
    return {
        "path_length": path_length,
        "velocity_rms": rms_norm(velocity),
        "acceleration_rms": rms_norm(acceleration),
        "jerk_rms": rms_norm(jerk),
        "roughness": (rms_norm(acceleration) + rms_norm(jerk))
        / (rms_norm(velocity) + 0.0001),
    }


def _aggregate_smoothness(actions: Sequence[Any]) -> dict[str, Any]:
    """Aggregate the FasterWAM UI smoothness definition across action chunks."""
    import numpy as np

    speeds: list[Any] = []
    accelerations: list[Any] = []
    jerks: list[Any] = []
    arm_roughness: dict[str, list[float]] = {"left": [], "right": []}
    chunks_analyzed = 0
    points_analyzed = 0
    for action in actions:
        values = np.asarray(action, dtype=np.float64)
        if values.ndim != 2:
            continue
        chunk_valid = False
        for arm, offset in (("left", 0), ("right", 8)):
            points = values[:, offset : offset + 3]
            if points.shape[0] < 4 or not np.isfinite(points).all():
                continue
            velocity = np.diff(points, axis=0)
            acceleration = np.diff(velocity, axis=0)
            jerk = np.diff(acceleration, axis=0)
            speeds.append(np.linalg.norm(velocity, axis=1))
            accelerations.append(np.linalg.norm(acceleration, axis=1))
            jerks.append(np.linalg.norm(jerk, axis=1))
            arm_roughness[arm].append(
                (_trajectory_smoothness(points))["roughness"]
            )
            points_analyzed += int(points.shape[0])
            chunk_valid = True
        if chunk_valid:
            chunks_analyzed += 1

    roughness_values = [
        value for values in arm_roughness.values() for value in values
    ]
    if not roughness_values:
        return {
            "chunks_analyzed": 0,
            "arm_trajectories_analyzed": 0,
            "points_analyzed": 0,
            "metrics": None,
            "arms": {},
        }

    def combined(samples: Sequence[Any]) -> Any:
        return np.concatenate(samples) if samples else np.asarray([], dtype=np.float64)

    def percentile(samples: Sequence[Any], value: float) -> float:
        values = combined(samples)
        return float(np.percentile(values, value)) if values.size else 0.0

    return {
        "chunks_analyzed": chunks_analyzed,
        "arm_trajectories_analyzed": len(roughness_values),
        "points_analyzed": points_analyzed,
        "metrics": {
            "roughness": float(np.mean(roughness_values)),
            "roughness_p95": float(np.percentile(roughness_values, 95)),
            "speed_median_mm_per_step": percentile(speeds, 50) * 1000.0,
            "acceleration_p95_mm_per_step2": percentile(accelerations, 95)
            * 1000.0,
            "jerk_p95_mm_per_step3": percentile(jerks, 95) * 1000.0,
        },
        "arms": {
            arm: {
                "roughness": float(np.mean(values)),
                "roughness_p95": float(np.percentile(values, 95)),
                "trajectories_analyzed": len(values),
            }
            for arm, values in arm_roughness.items()
            if values
        },
    }


def _read_action_artifact(
    output_root: Path,
    model: Mapping[str, Any],
) -> tuple[Any, Mapping[str, Any]]:
    import numpy as np

    relative_dir = Path(str(model.get("output_dir", "")))
    model_dir = (output_root / relative_dir).resolve()
    if not model_dir.is_relative_to(output_root):
        raise SuperOfflineError("comparison model output leaves its job directory")
    summary_path = model_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise SuperOfflineError("comparison summary has no action artifact")
    npz_name = str(artifacts.get("npz", ""))
    action_key = str(artifacts.get("canonical_action_key", ""))
    action_path = (model_dir / npz_name).resolve()
    if not action_path.is_relative_to(model_dir) or not action_path.is_file():
        raise SuperOfflineError("comparison action artifact is missing")
    with np.load(action_path, allow_pickle=False) as values:
        action = np.asarray(values[action_key], dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 16:
        raise SuperOfflineError("comparison action artifact has an invalid shape")
    return action, summary


class SuperOfflineService:
    """Thread-safe in-memory jobs for a pair-only comparison service."""

    def __init__(
        self,
        *,
        registry: SuperOfflineAdapterRegistry,
        runner: Any,
    ) -> None:
        self._registry = registry
        self._runner = runner
        self._lock = threading.Lock()
        self._jobs: dict[str, _ComparisonJob] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._max_concurrent_jobs = 1
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_concurrent_jobs,
            thread_name_prefix="SuperOffline",
        )

    def submit(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = self._validated_request(payload)
        warnings = self._request_warnings(request)
        job_id = uuid4().hex
        submitted_at = _utc_now()
        job = _ComparisonJob(
            job_id=job_id,
            request=replace(request, job_id=job_id),
            status="queued",
            submitted_at=submitted_at,
            heartbeat_at=submitted_at,
            warnings=warnings,
            logs=[
                {
                    "timestamp": submitted_at,
                    "stage": "queued",
                    "severity": "info",
                    "message": "Comparison accepted and queued",
                },
                *[
                    {
                        "timestamp": submitted_at,
                        "stage": "task_warning",
                        "severity": "warning",
                        "message": str(warning.get("message", "Task warning")),
                    }
                    for warning in warnings
                ],
            ],
        )
        with self._lock:
            pending = sum(
                job.status in {"queued", "running", "cancelling"}
                for job in self._jobs.values()
            )
            if pending >= 8:
                raise SuperOfflineCapacityError(
                    "SuperOffline comparison queue is full; wait for a job to finish"
                )
            self._jobs[job.job_id] = job
            self._cancel_events[job.job_id] = threading.Event()
        for warning in warnings:
            print(
                f"[SuperOffline {job_id[:8]}] WARNING task_warning: "
                f"{warning.get('message', 'Task warning')}",
                flush=True,
            )
        self._executor.submit(self._execute, job.job_id)
        return job.as_dict()

    def validate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = self._validated_request(payload)
        return {
            "valid": True,
            "request": request.as_dict(),
            "warnings": self._request_warnings(request),
        }

    def _validated_request(
        self, payload: Mapping[str, Any]
    ) -> SuperOfflineComparisonRequest:
        request = self._registry.parse_request(payload)
        validate_request = getattr(self._runner, "validate_request", None)
        if callable(validate_request):
            validate_request(request)
        return request

    def _request_warnings(
        self, request: SuperOfflineComparisonRequest
    ) -> list[dict[str, Any]]:
        warnings = getattr(self._runner, "request_warnings", None)
        if not callable(warnings):
            return []
        payload = warnings(request)
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise SuperOfflineError("runner request_warnings must return an array")
        return [dict(item) for item in payload if isinstance(item, Mapping)]

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                job = self._jobs[job_id]
            except KeyError as exc:
                raise KeyError(f"comparison job {job_id!r} was not found") from exc
            return job.as_dict()

    def list_jobs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 100))
        with self._lock:
            jobs = list(self._jobs.values())[-bounded_limit:]
        return [job.as_dict() for job in reversed(jobs)]

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                job = self._jobs[job_id]
            except KeyError as exc:
                raise KeyError(f"comparison job {job_id!r} was not found") from exc
            if job.status not in {"queued", "running", "cancelling"}:
                raise SuperOfflineConflictError(
                    f"job {job_id!r} is {job.status}; it cannot be cancelled"
                )
            cancel_event = self._cancel_events[job_id]
            cancel_event.set()
            timestamp = _utc_now()
            if job.status == "queued":
                job.status = "cancelled"
                job.stage = "cancelled"
                job.finished_at = timestamp
            else:
                job.status = "cancelling"
                job.stage = "cancelling"
            job.heartbeat_at = timestamp
            job.logs.append(
                {
                    "timestamp": timestamp,
                    "stage": "cancelling",
                    "severity": "warning",
                    "message": "Cancellation requested by operator",
                }
            )
            return job.as_dict()

    def retry(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                job = self._jobs[job_id]
            except KeyError as exc:
                raise KeyError(f"comparison job {job_id!r} was not found") from exc
            if job.status not in {"error", "cancelled"}:
                raise SuperOfflineConflictError(
                    f"job {job_id!r} is {job.status}; only failed jobs can be retried"
                )
            request = job.request
        models = []
        for model in request.models:
            payload: dict[str, Any] = {
                "adapter_id": model.adapter.id,
                "checkpoint": model.checkpoint,
                "config_name": model.config_name,
            }
            if model.variant_id is not None:
                payload["variant_id"] = model.variant_id
            if model.policy_params:
                payload["policy_params"] = dict(model.policy_params)
            models.append(payload)
        return self.submit(
            {
                "models": models,
                "observation_path": request.observation_path,
                "metadata": dict(request.metadata or {}),
            }
        )

    def execution_info(self) -> dict[str, Any]:
        metadata = getattr(self._runner, "public_metadata", None)
        payload: Mapping[str, Any] | None = None
        if callable(metadata):
            candidate = metadata()
            if isinstance(candidate, Mapping):
                payload = candidate
        info = dict(payload or {"mode": "injected-runner"})
        with self._lock:
            info["scheduler"] = {
                "max_concurrent_jobs": self._max_concurrent_jobs,
                "queued_jobs": sum(
                    job.status == "queued" for job in self._jobs.values()
                ),
                "running_jobs": sum(
                    job.status in {"running", "cancelling"}
                    for job in self._jobs.values()
                ),
            }
        return info

    def health_info(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "superoffline",
            "runtime_profile": os.environ.get("SUPERINFERENCE_RUNTIME_PROFILE"),
            "runtime_role": os.environ.get("SUPERINFERENCE_RUNTIME_ROLE"),
            "gpu_fault": gpu_safety.GPU_FAULT_LATCH.snapshot(),
            "scheduler": self.execution_info()["scheduler"],
        }

    def trajectory(self, job_id: str, *, sample_index: int = 0) -> dict[str, Any]:
        """Read the evaluator-declared canonical action artifacts for a result view."""
        job = self.get(job_id)
        if job.get("status") != "complete":
            raise SuperOfflineConflictError(
                f"job {job_id!r} has no completed trajectory result"
            )
        result = job.get("result") or {}
        samples = result.get("samples")
        if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)):
            if sample_index < 0 or sample_index >= len(samples):
                raise KeyError("comparison sample was not found")
            selected = samples[sample_index]
            if not isinstance(selected, Mapping):
                raise SuperOfflineError("comparison sample has an invalid result")
        else:
            selected = result
        output_root = Path(str(result.get("output_dir", ""))).expanduser().resolve()
        if not output_root.is_dir():
            raise KeyError(f"comparison output for job {job_id!r} was not found")
        models = selected.get("models")
        if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
            raise SuperOfflineError("comparison result has no model outputs")
        trajectories: list[dict[str, Any]] = []
        for model in models:
            if not isinstance(model, Mapping):
                continue
            try:
                action, _summary = _read_action_artifact(output_root, model)
            except ImportError as exc:
                raise SuperOfflineError(
                    "trajectory view requires NumPy in the registered runtime"
                ) from exc
            stride = max(1, int(action.shape[0] / 600))
            sampled = action[::stride]
            model_metadata = model.get("model")
            label = (
                str(model_metadata.get("label", "model"))
                if isinstance(model_metadata, Mapping)
                else "model"
            )
            trajectories.append(
                {
                    "label": label,
                    "left": sampled[:, :3].tolist(),
                    "right": sampled[:, 8:11].tolist(),
                    "steps": int(action.shape[0]),
                    "smoothness": {
                        "left": _trajectory_smoothness(action[:, :3]),
                        "right": _trajectory_smoothness(action[:, 8:11]),
                    },
                }
            )
        return {
            "job_id": job_id,
            "sample_index": sample_index,
            "trajectories": trajectories,
        }

    def smoothness(self, job_id: str) -> dict[str, Any]:
        """Aggregate per-checkpoint translation smoothness over every result chunk."""
        job = self.get(job_id)
        if job.get("status") != "complete":
            raise SuperOfflineConflictError(
                f"job {job_id!r} has no completed smoothness result"
            )
        result = job.get("result") or {}
        output_root = Path(str(result.get("output_dir", ""))).expanduser().resolve()
        if not output_root.is_dir():
            raise KeyError(f"comparison output for job {job_id!r} was not found")
        raw_samples = result.get("samples")
        samples = (
            list(raw_samples)
            if isinstance(raw_samples, Sequence)
            and not isinstance(raw_samples, (str, bytes))
            else [result]
        )
        first_models = samples[0].get("models", []) if samples else []
        if not isinstance(first_models, Sequence):
            raise SuperOfflineError("comparison result has no model outputs")
        colors = ("#4dd9ff", "#ff7a66", "#72e08a", "#f3c969")
        checkpoint_results: list[dict[str, Any]] = []
        for model_index, first_model in enumerate(first_models):
            if not isinstance(first_model, Mapping):
                continue
            actions: list[Any] = []
            skipped_chunks = 0
            for sample in samples:
                models = sample.get("models", []) if isinstance(sample, Mapping) else []
                if not isinstance(models, Sequence) or model_index >= len(models):
                    skipped_chunks += 1
                    continue
                model = models[model_index]
                if not isinstance(model, Mapping):
                    skipped_chunks += 1
                    continue
                try:
                    action, _summary = _read_action_artifact(output_root, model)
                except (OSError, KeyError, ValueError, SuperOfflineError):
                    skipped_chunks += 1
                    continue
                actions.append(action)
            analysis = _aggregate_smoothness(actions)
            skipped_chunks += len(actions) - int(analysis["chunks_analyzed"])
            metadata = first_model.get("model")
            model_metadata = metadata if isinstance(metadata, Mapping) else {}
            checkpoint_results.append(
                {
                    "backend": str(model_metadata.get("adapter_id", "model")),
                    "checkpoint": str(model_metadata.get("checkpoint", "")),
                    "checkpoint_name": str(
                        model_metadata.get("label")
                        or model_metadata.get("config_name")
                        or f"checkpoint {model_index + 1}"
                    ),
                    "color": colors[model_index % len(colors)],
                    "skipped_chunks": skipped_chunks,
                    **analysis,
                }
            )
        ranked = sorted(
            (
                (float(item["metrics"]["roughness"]), index)
                for index, item in enumerate(checkpoint_results)
                if item.get("metrics") is not None
            ),
            key=lambda item: (item[0], item[1]),
        )
        for rank, (_value, index) in enumerate(ranked, start=1):
            checkpoint_results[index]["rank"] = rank
        return {
            "schema": "superoffline.chunk_smoothness",
            "job_id": job_id,
            "observation_count": len(samples),
            "derivative_interval": "one_predicted_action_step",
            "position_units": "meters",
            "checkpoints": checkpoint_results,
        }

    def shutdown(self) -> None:
        with self._lock:
            for cancel_event in self._cancel_events.values():
                cancel_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _execute(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.status != "queued":
                return
            job.status = "running"
            job.stage = "start"
            job.progress = 1
            job.started_at = _utc_now()
            job.heartbeat_at = job.started_at
            job.logs.append(
                {
                    "timestamp": job.started_at,
                    "stage": "start",
                    "severity": "info",
                    "message": "Runtime profile comparison started",
                }
            )
            request = job.request
            cancel_event = self._cancel_events[job_id]

        def handle_event(event: Mapping[str, Any]) -> None:
            timestamp = _utc_now()
            stage = str(event.get("stage", "runtime"))[:80]
            message = str(event.get("message", ""))[:1200]
            severity = str(event.get("severity", "info"))[:20]
            with self._lock:
                current = self._jobs[job_id]
                current.heartbeat_at = timestamp
                current.stage = stage
                progress = event.get("progress")
                if progress is not None:
                    try:
                        current.progress = max(
                            current.progress,
                            min(99, max(0, int(progress))),
                        )
                    except (TypeError, ValueError):
                        pass
                if event.get("latest_checkpoint") is not None:
                    current.latest_checkpoint = str(event["latest_checkpoint"])
                if event.get("latest_observation") is not None:
                    current.latest_observation = str(event["latest_observation"])
                if message:
                    current.logs.append(
                        {
                            "timestamp": timestamp,
                            "stage": stage,
                            "severity": severity,
                            "message": message,
                        }
                    )
                    current.logs = current.logs[-500:]
            if message:
                level = "WARNING " if severity == "warning" else ""
                print(
                    f"[SuperOffline {job_id[:8]}] {level}{stage}: {message}",
                    flush=True,
                )
        try:
            result = _run_runner(
                self._runner,
                request,
                event_callback=handle_event,
                cancel_event=cancel_event,
            )
            # Fail inside the job instead of surfacing an invalid result through
            # the status endpoint later.
            json.dumps(result)
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                if cancel_event.is_set():
                    job.status = "cancelled"
                    job.stage = "cancelled"
                    job.error = None
                    message = "Comparison cancelled"
                    severity = "warning"
                else:
                    job.status = "error"
                    job.stage = "fatal_error"
                    job.error = f"{type(exc).__name__}: {exc}"
                    message = job.error
                    severity = "error"
                job.finished_at = _utc_now()
                job.heartbeat_at = job.finished_at
                job.logs.append(
                    {
                        "timestamp": job.finished_at,
                        "stage": job.stage,
                        "severity": severity,
                        "message": message,
                    }
                )
        else:
            with self._lock:
                job = self._jobs[job_id]
                if cancel_event.is_set():
                    job.status = "cancelled"
                    job.stage = "cancelled"
                    message = "Comparison cancelled"
                else:
                    job.status = "complete"
                    job.stage = "complete"
                    job.progress = 100
                    job.result = result
                    message = (
                        f"Comparison completed with "
                        f"{int(result.get('observation_count', 0))} observation chunk(s)"
                    )
                job.finished_at = _utc_now()
                job.heartbeat_at = job.finished_at
                job.logs.append(
                    {
                        "timestamp": job.finished_at,
                        "stage": job.stage,
                        "severity": "info",
                        "message": message,
                    }
                )


def _json_response(
    start_response: Callable[..., Any],
    status: str,
    payload: Mapping[str, Any] | Sequence[Any],
) -> list[bytes]:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    start_response(
        status,
        [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "no-store"),
        ],
    )
    return [body]


def _html_response(start_response: Callable[..., Any]) -> list[bytes]:
    body = _PAGE.encode("utf-8")
    start_response(
        "200 OK",
        [
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "no-store"),
        ],
    )
    return [body]


def _query(environ: Mapping[str, Any]) -> dict[str, str]:
    raw = str(environ.get("QUERY_STRING", ""))
    return {
        key: values[-1]
        for key, values in parse_qs(raw, keep_blank_values=True).items()
        if values
    }


def _observation_runs(base_dir: Path) -> list[dict[str, Any]]:
    """List finalized recorder runs without importing the model runtimes."""
    base = base_dir.expanduser().resolve()
    if not base.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(base.glob("**/manifest.json"), reverse=True):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if manifest.get("schema") != "superinference.timestamped_observation":
            continue
        if not manifest.get("stopped_at_ns"):
            continue
        rows.append(
            {
                "path": str(manifest_path.parent),
                "source_type": "recorder",
                "source_label": "Observation recorder",
                "run_started_at_ns": manifest.get("run_started_at_ns"),
                "stopped_at_ns": manifest.get("stopped_at_ns"),
                "frame_count": manifest.get("frame_count"),
                "inference_count": manifest.get("inference_count"),
                "source_profile": manifest.get("source_profile"),
            }
        )
    return rows


def _observation_inputs(*base_dirs: Path) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for base_dir in base_dirs:
        for item in _observation_runs(base_dir):
            unique[item["path"]] = item
    return sorted(unique.values(), key=lambda item: item["path"], reverse=True)


def _observation_samples(root: Path) -> list[dict[str, Any]]:
    root = root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    index_name = str(manifest.get("inference_index", "inference.jsonl"))
    index_path = root / Path(index_name)
    if Path(index_name).name != index_name or not index_path.is_file():
        return []
    samples: list[dict[str, Any]] = []
    try:
        lines = index_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, Mapping):
            continue
        artifact = str(item.get("file", ""))
        relative = Path(artifact)
        if (
            not artifact
            or relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != "inference"
        ):
            continue
        samples.append(
            {
                "number": number,
                "timestamp_ns": item.get("timestamp_ns"),
                "file": artifact,
                "path": str((root / relative).resolve()),
                "kind": item.get("kind", "inference"),
            }
        )
    return samples


def _observation_preview(root: Path, index: int, camera_index: int) -> bytes:
    samples = _observation_samples(root)
    if index < 0 or index >= len(samples):
        raise KeyError("observation sample was not found")
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise SuperOfflineError("image preview requires NumPy and Pillow in the registered runtime") from exc
    sample = samples[index]
    artifact = Path(sample["path"]).resolve()
    if not artifact.is_relative_to(root.expanduser().resolve()) or not artifact.is_file():
        raise KeyError("observation artifact was not found")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    index_name = str(manifest.get("inference_index", "inference.jsonl"))
    rows = (root / index_name).read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[int(sample["number"])])
    keys = row.get("keys", []) if isinstance(row, Mapping) else []
    candidates: list[str] = []
    with np.load(artifact, allow_pickle=False) as values:
        for item in keys:
            if not isinstance(item, Mapping):
                continue
            array_key = str(item.get("array_key", ""))
            if array_key in values and np.asarray(values[array_key]).ndim >= 3:
                candidates.append(array_key)
        if not candidates:
            candidates = [key for key in values.files if np.asarray(values[key]).ndim >= 3]
        if camera_index >= len(candidates):
            raise KeyError("camera preview was not found")
        array = np.asarray(values[candidates[min(camera_index, len(candidates) - 1)]])
    if array.ndim == 4 and array.shape[0] in {2, 3, 4}:
        array = array[camera_index % array.shape[0]]
    while array.ndim > 3:
        array = array[0]
    if array.ndim != 3:
        raise SuperOfflineError("recorded camera array has an invalid shape")
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] > 3:
        array = array[..., :3]
    array = np.asarray(array, dtype=np.float32)
    if (float(array.max()) if array.size else 0.0) <= 1.5:
        array *= 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    output = BytesIO()
    image.save(output, format="JPEG", quality=88)
    return output.getvalue()


def _binary_response(
    start_response: Callable[..., Any],
    status: str,
    body: bytes,
    content_type: str,
) -> list[bytes]:
    start_response(
        status,
        [
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "no-store"),
        ],
    )
    return [body]


def _gpu_status() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    if completed.returncode != 0:
        return {
            "available": False,
            "error": completed.stderr.strip() or "nvidia-smi failed",
        }
    gpus: list[dict[str, Any]] = []
    for row in completed.stdout.splitlines():
        fields = [field.strip() for field in row.split(",", 3)]
        if len(fields) != 4:
            continue
        try:
            gpus.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "memory_used_mib": int(fields[2]),
                    "memory_total_mib": int(fields[3]),
                }
            )
        except ValueError:
            continue
    return {
        "available": bool(gpus),
        "gpus": gpus,
        "fault": gpu_safety.GPU_FAULT_LATCH.snapshot(),
    }


def _infer_discovered_fastwam_variant(checkpoint: Path, config_path: Path) -> str:
    """Classify a discovered checkpoint without importing a model runtime."""
    try:
        text = config_path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        text = ""
    hint = str(checkpoint).lower()
    if (
        "one_pass_future_cache" in text
        or "fasterwam" in hint
        or re.search(r"(?:^|[/_.-])c4(?:$|[/_.-])", hint)
    ):
        return "fasterwam"
    if (
        re.search(r"condition_history_frames\s*:\s*4\b", text)
        or re.search(r"obs_horizon\s*:\s*4\b", text)
        or "fastwam-memory" in hint
        or "fastwam_memory" in hint
        or "b1_1_s4" in hint
    ):
        return "fastwam-memory"
    return "fastwam"


_GENERIC_CHECKPOINT_LEAF = re.compile(r"^(?:step[_-]?)?\d+$", re.IGNORECASE)


def _checkpoint_display_name(root: Path, checkpoint: Path) -> str:
    """Name an experiment, not its generic numeric/step checkpoint leaf."""
    try:
        relative = checkpoint.relative_to(root)
    except ValueError:
        relative = Path(checkpoint.name)
    parts = list(relative.parts)
    if len(parts) > 1 and _GENERIC_CHECKPOINT_LEAF.fullmatch(parts[-1]):
        parts.pop()
    return "/".join(parts) or checkpoint.name


def _checkpoint_catalog(
    fastwam_root: str | None,
    openpi_root: str | None,
) -> list[dict[str, Any]]:
    roots: list[Path] = []
    raw_fastwam = fastwam_root or str(REPO_ROOT / "checkpoints")
    for item in raw_fastwam.replace(",", os.pathsep).split(os.pathsep):
        if item.strip():
            roots.append(Path(item.strip()).expanduser().resolve())
    records: dict[str, dict[str, Any]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for config in root.glob("**/config.yaml"):
            checkpoint = config.parent.resolve()
            variant_id = _infer_discovered_fastwam_variant(checkpoint, config)
            records[str(checkpoint)] = {
                "backend": "fasterwam" if variant_id == "fasterwam" else "fastwam",
                "variant_id": variant_id,
                "path": str(checkpoint),
                "name": checkpoint.name,
                "display_name": _checkpoint_display_name(root, checkpoint),
                "config_path": str(config),
            }
    openpi = Path(openpi_root or str(REPO_ROOT / "checkpoints" / "openpi")).expanduser().resolve()
    if openpi.is_dir():
        for params in openpi.glob("**/params"):
            if params.is_dir():
                checkpoint = params.parent.resolve()
                try:
                    assets = discover_openpi_norm_stats_assets(checkpoint)
                except OpenPiCheckpointContractError:
                    continue
                records[str(checkpoint)] = {
                    "backend": "openpi",
                    "path": str(checkpoint),
                    "name": checkpoint.name,
                    "display_name": _checkpoint_display_name(openpi, checkpoint),
                    "params_path": str(params),
                    "pi_config_name": "default",
                    "asset_id": assets["asset_id"],
                    "asset_ids": assets["asset_ids"],
                }
    display_counts = Counter(
        (item["backend"], item["display_name"]) for item in records.values()
    )
    for item in records.values():
        display_key = (item["backend"], item["display_name"])
        if display_counts[display_key] > 1:
            item["display_name"] = f"{item['display_name']} ({Path(item['path']).name})"
    return sorted(
        records.values(),
        key=lambda item: (
            item["backend"],
            str(item.get("display_name", item["name"])).lower(),
            item["path"],
        ),
    )


def _request_json(environ: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_length = str(environ.get("CONTENT_LENGTH", "")).strip()
    if not raw_length:
        raise SuperOfflineHttpError("Content-Length is required")
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise SuperOfflineHttpError("Content-Length must be an integer") from exc
    if length < 0 or length > _MAX_REQUEST_BYTES:
        raise SuperOfflineHttpError("request body is too large")
    raw_body = environ["wsgi.input"].read(length)
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SuperOfflineHttpError("request body must be UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise SuperOfflineHttpError("request body must be a JSON object")
    return payload


class SuperOfflineApplication:
    """Dependency-free WSGI application for local comparison jobs."""

    def __init__(
        self,
        *,
        registry: SuperOfflineAdapterRegistry,
        service: SuperOfflineService,
    ) -> None:
        self.registry = registry
        self.service = service

    def __call__(
        self,
        environ: Mapping[str, Any],
        start_response: Callable[..., Any],
    ) -> list[bytes]:
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        path = str(environ.get("PATH_INFO", "/"))
        try:
            if method == "GET" and path == "/":
                return _html_response(start_response)
            if method == "GET" and path == "/plotly.min.js":
                try:
                    import plotly
                except ImportError as exc:
                    raise SuperOfflineError(
                        "Plotly is required for the interactive trajectory view"
                    ) from exc
                bundle = (
                    Path(plotly.__file__).resolve().parent
                    / "package_data"
                    / "plotly.min.js"
                )
                if not bundle.is_file():
                    raise KeyError("Plotly browser bundle was not found")
                return _binary_response(
                    start_response,
                    "200 OK",
                    bundle.read_bytes(),
                    "application/javascript",
                )
            if method == "GET" and path == "/api/models":
                return _json_response(
                    start_response,
                    "200 OK",
                    {"models": self.registry.public_models()},
                )
            if method == "GET" and path == "/api/service-info":
                return _json_response(
                    start_response,
                    "200 OK",
                    self.service.execution_info(),
                )
            if method == "GET" and path == "/api/healthz":
                return _json_response(
                    start_response,
                    "200 OK",
                    self.service.health_info(),
                )
            if method == "GET" and path == "/api/gpu":
                return _json_response(start_response, "200 OK", _gpu_status())
            if method == "GET" and path == "/api/observations":
                query = _query(environ)
                if query.get("base_dir"):
                    inputs = _observation_inputs(Path(query["base_dir"]))
                else:
                    inputs = _observation_inputs(
                        REPO_ROOT / "logs" / "offline_observations",
                        REPO_ROOT / "logs" / "fastwam_observations",
                    )
                return _json_response(
                    start_response,
                    "200 OK",
                    inputs,
                )
            if method == "GET" and path == "/api/observations/samples":
                observation_path = _query(environ).get("path", "")
                if not observation_path:
                    raise SuperOfflineHttpError("path is required")
                return _json_response(
                    start_response,
                    "200 OK",
                    _observation_samples(Path(unquote(observation_path))),
                )
            if method == "GET" and path == "/api/observation/preview":
                query = _query(environ)
                observation_path = query.get("path", "")
                if not observation_path:
                    raise SuperOfflineHttpError("path is required")
                try:
                    sample_index = int(query.get("inference_index", "0"))
                    camera_index = int(query.get("camera_index", "0"))
                except ValueError as exc:
                    raise SuperOfflineHttpError("preview indexes must be integers") from exc
                return _binary_response(
                    start_response,
                    "200 OK",
                    _observation_preview(
                        Path(unquote(observation_path)), sample_index, camera_index
                    ),
                    "image/jpeg",
                )
            if method == "GET" and path == "/api/checkpoints":
                query = _query(environ)
                return _json_response(
                    start_response,
                    "200 OK",
                    _checkpoint_catalog(
                        query.get("fastwam_dir"),
                        query.get("openpi_dir"),
                    ),
                )
            if method == "POST" and path == "/api/validate":
                return _json_response(
                    start_response,
                    "200 OK",
                    self.service.validate(_request_json(environ)),
                )
            if method == "POST" and path == "/api/compare":
                job = self.service.submit(_request_json(environ))
                return _json_response(start_response, "202 Accepted", job)
            if method == "GET" and path == "/api/jobs":
                return _json_response(
                    start_response,
                    "200 OK",
                    {"jobs": self.service.list_jobs()},
                )
            if method == "GET" and path.startswith("/api/jobs/"):
                if "/artifact/" in path:
                    prefix, relative_name = path.split("/artifact/", 1)
                    job_id = prefix.removeprefix("/api/jobs/")
                    job = self.service.get(job_id)
                    result = job.get("result") or {}
                    output_root = Path(str(result.get("output_dir", ""))).expanduser().resolve()
                    artifact = (output_root / unquote(relative_name)).resolve()
                    if not artifact.is_relative_to(output_root) or not artifact.is_file():
                        raise KeyError("comparison artifact was not found")
                    content_type = mimetypes.guess_type(artifact.name)[0] or "application/octet-stream"
                    return _binary_response(
                        start_response,
                        "200 OK",
                        artifact.read_bytes(),
                        content_type,
                    )
                if path.endswith("/trajectory"):
                    job_id = path.removeprefix("/api/jobs/").removesuffix("/trajectory").strip("/")
                    if not job_id or "/" in job_id:
                        raise KeyError("comparison job was not found")
                    try:
                        sample_index = int(_query(environ).get("sample", "0"))
                    except ValueError as exc:
                        raise SuperOfflineHttpError("sample must be an integer") from exc
                    return _json_response(
                        start_response,
                        "200 OK",
                        self.service.trajectory(
                            job_id,
                            sample_index=sample_index,
                        ),
                    )
                if path.endswith("/smoothness"):
                    job_id = (
                        path.removeprefix("/api/jobs/")
                        .removesuffix("/smoothness")
                        .strip("/")
                    )
                    if not job_id or "/" in job_id:
                        raise KeyError("comparison job was not found")
                    return _json_response(
                        start_response,
                        "200 OK",
                        self.service.smoothness(job_id),
                    )
                job_id = path.removeprefix("/api/jobs/")
                if not job_id or "/" in job_id:
                    return _json_response(
                        start_response,
                        "404 Not Found",
                        {"error": "comparison job was not found"},
                    )
                return _json_response(start_response, "200 OK", self.service.get(job_id))
            if method == "POST" and path.startswith("/api/jobs/"):
                parts = path.split("/")
                if len(parts) == 5 and parts[4] in {"cancel", "retry"}:
                    job_id = parts[3]
                    result = (
                        self.service.cancel(job_id)
                        if parts[4] == "cancel"
                        else self.service.retry(job_id)
                    )
                    return _json_response(start_response, "200 OK", result)
        except SuperOfflineCapacityError as exc:
            return _json_response(start_response, "503 Service Unavailable", {"error": str(exc)})
        except SuperOfflineConflictError as exc:
            return _json_response(start_response, "409 Conflict", {"error": str(exc)})
        except SuperOfflineError as exc:
            return _json_response(start_response, "400 Bad Request", {"error": str(exc)})
        except KeyError as exc:
            return _json_response(start_response, "404 Not Found", {"error": str(exc)})
        except Exception as exc:
            return _json_response(
                start_response,
                "500 Internal Server Error",
                {"error": f"internal SuperOffline service error: {type(exc).__name__}"},
            )
        return _json_response(
            start_response,
            "404 Not Found",
            {"error": "route was not found"},
        )


def create_app(
    *,
    registry: SuperOfflineAdapterRegistry = SUPER_OFFLINE_ADAPTERS,
    runner: Any | None = None,
    executor: ComparisonExecutor | None = None,
) -> SuperOfflineApplication:
    """Create a local app; tests can inject a runner without model packages."""
    if runner is not None and executor is not None:
        raise ValueError("pass either runner or executor, not both")
    if runner is not None:
        selected_runner = runner
    else:
        selected_executor = executor
        if selected_executor is None:
            selected_executor = RealRobotOfflineComparisonExecutor(repo_root=REPO_ROOT)
        selected_runner = RuntimeProfileComparisonRunner(
            selected_executor,
            repo_root=REPO_ROOT,
        )
    return SuperOfflineApplication(
        registry=registry,
        service=SuperOfflineService(registry=registry, runner=selected_runner),
    )


def _load_executor(reference: str) -> ComparisonExecutor:
    module_name, separator, attribute_name = reference.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("--executor must use module:object syntax")
    module = importlib.import_module(module_name)
    executor = getattr(module, attribute_name, None)
    validate_request = getattr(executor, "validate_request", None)
    run = getattr(executor, "run", None)
    if not callable(validate_request) or not callable(run):
        raise ValueError(
            f"executor {reference!r} must expose validate_request(request) and "
            "run(request, contexts)"
        )
    return executor


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8767, type=int)
    parser.add_argument(
        "--executor",
        default=None,
        help="module:object replacement executor for registered comparison pairs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        executor = _load_executor(args.executor) if args.executor else None
    except (ImportError, ValueError, AttributeError) as exc:
        print(f"Could not load SuperOffline executor: {exc}", file=sys.stderr)
        return 2
    app = create_app(executor=executor)
    try:
        with make_server(args.host, args.port, app) as server:
            print(f"SuperOffline is listening at http://{args.host}:{args.port}")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                return 0
    finally:
        app.service.shutdown()
    return 0


_LEGACY_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SuperOffline</title>
  <style>
    :root { color: #17232e; background: #f5f7f7; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 320px; }
    header { background: #153d4a; color: #fff; padding: 18px 24px; font-size: 20px; font-weight: 700; }
    main { max-width: 1120px; margin: 0 auto; padding: 4px 24px 24px; }
    .source, .models, .result { border-bottom: 1px solid #cbd6da; padding: 20px 0; }
    .service-status { display: flex; align-items: center; justify-content: space-between; gap: 16px; border-bottom: 1px solid #cbd6da; padding: 16px 0; color: #41535b; font-size: 13px; }
    .service-status strong { color: #17232e; }
    .service-status.error { color: #a93328; }
    .model-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .model { border: 1px solid #d7e0e3; border-top: 4px solid #d9783e; padding: 16px; min-width: 0; }
    h1, h2 { margin: 0; font-size: 16px; }
    .model h2 { margin-bottom: 14px; }
    label { display: block; color: #3c4b54; font-size: 13px; font-weight: 650; margin-top: 12px; }
    input, select, button { font: inherit; }
    input, select { width: 100%; border: 1px solid #9fb0b7; border-radius: 4px; min-height: 38px; padding: 7px 9px; background: #fff; color: #17232e; }
    input[readonly] { background: #f1f5f5; color: #53636b; }
    .config-name { display: block; min-height: 38px; border: 1px solid #9fb0b7; border-radius: 4px; padding: 8px 9px; background: #f1f5f5; color: #53636b; font: 12px/1.35 ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap: anywhere; }
    .actions { display: flex; justify-content: flex-end; gap: 10px; margin-top: 18px; }
    .secondary { background: #fff; color: #0e6874; }
    .job-history { border-bottom: 1px solid #cbd6da; padding: 20px 0; }
    .job-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; border-top: 1px solid #d7e0e3; padding: 10px 0; font-size: 12px; }
    .job-row code { color: #53636b; }
    .job-row button { min-height: 30px; padding: 4px 10px; font-size: 12px; }
    .job-actions { display: flex; gap: 6px; }
    button { min-height: 38px; border: 1px solid #0e6874; border-radius: 4px; padding: 7px 16px; background: #0e6874; color: #fff; cursor: pointer; }
    button:disabled { background: #98a9ae; border-color: #98a9ae; cursor: wait; }
    .status { min-height: 24px; margin: 14px 0 0; color: #41535b; white-space: pre-wrap; }
    .status.error { color: #a93328; }
    pre { margin: 12px 0 0; max-height: 360px; overflow: auto; border: 1px solid #d7e0e3; padding: 12px; background: #f8fafb; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
    @media (max-width: 700px) { main { padding: 4px 14px 20px; } header { padding: 15px; } .model-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>SuperOffline</header>
  <main>
    <section class="service-status" aria-label="Service status">
      <strong>Service</strong>
      <span id="service-health">Checking...</span>
    </section>
    <section class="source" aria-label="Observation input">
      <label for="observation-path">Recorded observation path</label>
      <input id="observation-path" autocomplete="off" spellcheck="false" placeholder="logs/.../run_...">
      <label for="prompt">Task prompt</label>
      <input id="prompt" autocomplete="off" spellcheck="false" placeholder="Put gift into box and cover lid">
      <div class="actions">
        <button id="validate" class="secondary" type="button">Validate input</button>
      </div>
    </section>
    <section class="models" aria-label="Comparison models">
      <div class="model-grid" id="model-grid"></div>
      <div class="actions"><button id="compare" type="button" disabled>Compare</button></div>
      <div class="status" id="status" role="status"></div>
    </section>
    <section class="job-history" aria-label="Job history">
      <h1>Recent jobs</h1>
      <div id="job-history-list">No jobs yet.</div>
    </section>
    <section class="result" aria-label="Comparison result">
      <h1>Job</h1>
      <pre id="result">Waiting for a comparison.</pre>
    </section>
  </main>
  <script>
    const state = [
      { adapterId: '', variantId: '', checkpoint: '', configName: '' },
      { adapterId: '', variantId: '', checkpoint: '', configName: '' },
    ];
    let catalog = [];
    let executionInfo = {};
    let serviceHealth = {};
    let activeJob = null;
    let validationReady = false;
    const grid = document.getElementById('model-grid');
    const status = document.getElementById('status');
    const compareButton = document.getElementById('compare');
    const validateButton = document.getElementById('validate');
    const result = document.getElementById('result');
    const serviceHealthNode = document.getElementById('service-health');
    const jobHistoryList = document.getElementById('job-history-list');

    function element(tag, attributes = {}, text = '') {
      const node = document.createElement(tag);
      Object.entries(attributes).forEach(([key, value]) => { node[key] = value; });
      if (text) node.textContent = text;
      return node;
    }
    function adapterById(id) { return catalog.find(item => item.adapter_id === id); }
    function setStatus(message, isError = false) {
      status.textContent = message;
      status.className = isError ? 'status error' : 'status';
    }
    function invalidateValidation() {
      validationReady = false;
      updateExecutionAvailability();
    }
    function renderServiceHealth() {
      const scheduler = serviceHealth.scheduler || {};
      const running = Number(scheduler.running_jobs || 0);
      const queued = Number(scheduler.queued_jobs || 0);
      serviceHealthNode.textContent = serviceHealth.status === 'ok'
        ? `Ready · ${running} running · ${queued} queued`
        : (serviceHealth.error || 'Unavailable');
      serviceHealthNode.parentElement.className = serviceHealth.status === 'ok'
        ? 'service-status'
        : 'service-status error';
    }
    function currentAdapter(index) {
      const configured = adapterById(state[index].adapterId);
      return configured || catalog.find(item => item.available) || null;
    }
    function option(select, value, text, selected = false) {
      const node = element('option', { value }, text);
      node.selected = selected;
      select.append(node);
    }
    function field(container, labelText, control) {
      const label = element('label', {}, labelText);
      label.append(control);
      container.append(label);
    }
    function selectedContract(index) {
      const adapter = adapterById(state[index].adapterId);
      if (!adapter) return null;
      if (adapter.variant_family) {
        const variant = (adapter.variants || []).find(item => item.id === state[index].variantId);
        return variant || null;
      }
      return (adapter.config_contracts || []).find(item => item.config_name === state[index].configName) || null;
    }
    function pairIsCompatible() {
      const first = selectedContract(0);
      const second = selectedContract(1);
      if (!first || !second) return false;
      return first.action_frame_id === second.action_frame_id
        && first.action_units === second.action_units;
    }
    function pairHasSharedReviewedSource() {
      const first = selectedContract(0);
      const second = selectedContract(1);
      if (!first || !second) return false;
      const firstProfiles = Array.isArray(first.source_profiles) ? first.source_profiles : [];
      const secondProfiles = Array.isArray(second.source_profiles) ? second.source_profiles : [];
      return firstProfiles.some(profile => secondProfiles.includes(profile));
    }
    function unavailableSelectedAdapter() {
      return state
        .map(model => adapterById(model.adapterId))
        .find(adapter => adapter && adapter.available === false) || null;
    }
    function pairIsSupportedByExecutor() {
      const supported = Array.isArray(executionInfo.supported_adapter_ids)
        ? executionInfo.supported_adapter_ids
        : [];
      const requiresSameAdapter = executionInfo.requires_same_adapter === true;
      const selected = state.map(model => {
        const adapter = adapterById(model.adapterId);
        return adapter ? adapter.adapter_id : '';
      });
      return selected.every(adapterId => supported.includes(adapterId))
        && (!requiresSameAdapter || selected[0] === selected[1]);
    }
    function updateExecutionAvailability() {
      const hasAvailableAdapter = catalog.some(item => item.available);
      const builtin = executionInfo.mode === 'builtin-real-robot-offline';
      const compatible = pairIsCompatible();
      const unavailable = unavailableSelectedAdapter();
      const executable = !unavailable
        && (!builtin || pairIsSupportedByExecutor())
        && compatible;
      compareButton.disabled = !hasAvailableAdapter || !executable || !validationReady;
      if (unavailable) {
        setStatus(unavailable.availability_error || 'The selected model is unavailable for comparison.', true);
      } else if (!compatible) {
        setStatus('These models do not share an action-frame contract.', true);
      } else if (builtin && !pairIsSupportedByExecutor()) {
        setStatus('The selected pair has no built-in read-only evaluator.', true);
      } else if (!pairHasSharedReviewedSource()) {
        setStatus('The selected models require different reviewed recording sources. Recorder preflight will require a reviewed conversion before execution.', true);
      } else if (builtin) {
        setStatus('Ready for a read-only real-robot comparison.');
      }
    }
    function renderModels() {
      grid.replaceChildren();
      state.forEach((model, index) => {
        const card = element('article', { className: 'model' });
        card.append(element('h2', {}, index === 0 ? 'Model A' : 'Model B'));
        const adapterSelect = element('select', { ariaLabel: `Model ${index + 1} adapter` });
        catalog.forEach(adapter => option(
          adapterSelect,
          adapter.adapter_id,
          adapter.available ? adapter.label : `${adapter.label} (unavailable)`,
          adapter.adapter_id === (model.adapterId || (currentAdapter(index) || {}).adapter_id),
        ));
        adapterSelect.addEventListener('change', () => {
          invalidateValidation();
          model.adapterId = adapterSelect.value;
          model.variantId = '';
          model.configName = '';
          renderModels();
        });
        field(card, 'Adapter', adapterSelect);
        const adapter = currentAdapter(index);
        if (!adapter) { card.append(element('p', {}, 'No real-robot adapter is available.')); grid.append(card); return; }
        model.adapterId = adapter.adapter_id;
        const adapterAvailable = adapter.available !== false;
        if (adapter.variant_family) {
          const variantSelect = element('select', { ariaLabel: `Model ${index + 1} FastWAM variant` });
          const variants = adapter.variants || [];
          variants.forEach(variant => option(
            variantSelect,
            variant.id,
            variant.label,
            variant.id === (model.variantId || (variants[0] || {}).id),
          ));
          variantSelect.disabled = variants.length === 0;
          variantSelect.addEventListener('change', () => { invalidateValidation(); model.variantId = variantSelect.value; renderModels(); });
          field(card, 'Variant', variantSelect);
          const variant = variants.find(item => item.id === (model.variantId || variantSelect.value));
          model.variantId = variant ? variant.id : '';
          const config = element('output', { className: 'config-name' }, variant ? variant.config_name : '');
          field(card, 'Config', config);
          const source = element('output', { className: 'config-name' }, variant ? (variant.source_profiles || []).join(', ') : '');
          field(card, 'Reviewed recording source', source);
          if (!variants.length && adapter.availability_error) card.append(element('p', {}, adapter.availability_error));
        } else {
          const configNames = Array.isArray(adapter.allowed_config_names)
            ? adapter.allowed_config_names
            : [];
          let config;
          if (configNames.length) {
            config = element('select', { ariaLabel: `Model ${index + 1} config` });
            configNames.forEach(configName => option(
              config,
              configName,
              configName,
              configName === (model.configName || adapter.default_config_name || configNames[0]),
            ));
            model.configName = config.value;
            config.addEventListener('change', () => { invalidateValidation(); model.configName = config.value; renderModels(); });
            config.disabled = !adapterAvailable;
          } else {
            config = element('input', { value: model.configName || adapter.default_config_name || '', autocomplete: 'off', spellcheck: false });
            model.configName = config.value;
            config.addEventListener('input', () => { model.configName = config.value; });
            config.disabled = !adapterAvailable;
          }
          field(card, 'Config', config);
          const selected = (adapter.config_contracts || []).find(item => item.config_name === model.configName);
          const source = element('output', { className: 'config-name' }, selected ? (selected.source_profiles || []).join(', ') : '');
          field(card, 'Reviewed recording source', source);
          if (!adapterAvailable && adapter.availability_error) card.append(element('p', {}, adapter.availability_error));
        }
        const checkpoint = element('input', { value: model.checkpoint, autocomplete: 'off', spellcheck: false, placeholder: 'checkpoint directory or file' });
        checkpoint.addEventListener('input', () => { invalidateValidation(); model.checkpoint = checkpoint.value; });
        checkpoint.disabled = !adapterAvailable;
        field(card, 'Checkpoint', checkpoint);
        grid.append(card);
      });
      updateExecutionAvailability();
    }
    function requestModels() {
      return Promise.all([
        fetch('/api/models'),
        fetch('/api/service-info'),
      ]).then(async ([modelsResponse, infoResponse]) => {
        const payload = await modelsResponse.json();
        const info = await infoResponse.json();
        if (!modelsResponse.ok) throw Error(payload.error || 'Could not load models');
        if (!infoResponse.ok) throw Error(info.error || 'Could not load service state');
        catalog = payload.models || [];
        executionInfo = info;
        renderModels();
      });
    }
    function requestModel(model) {
      const adapter = adapterById(model.adapterId);
      const payload = { adapter_id: model.adapterId, checkpoint: model.checkpoint.trim() };
      if (adapter && adapter.variant_family) payload.variant_id = model.variantId;
      else payload.config_name = model.configName.trim();
      return payload;
    }
    function requestPayload() {
      const payload = {
        observation_path: document.getElementById('observation-path').value.trim(),
        models: state.map(requestModel),
      };
      const prompt = document.getElementById('prompt').value.trim();
      if (prompt) payload.metadata = { prompt };
      return payload;
    }
    function renderJobHistory(jobs) {
      jobHistoryList.replaceChildren();
      if (!jobs.length) {
        jobHistoryList.textContent = 'No jobs yet.';
        return;
      }
      jobs.forEach(job => {
        const row = element('div', { className: 'job-row' });
        const label = element('span', {}, `${job.status} · ${job.job_id.slice(0, 8)}`);
        const actions = element('span', { className: 'job-actions' });
        const view = element('button', { className: 'secondary', type: 'button' }, 'View');
        view.addEventListener('click', () => { activeJob = job.job_id; result.textContent = JSON.stringify(job, null, 2); });
        actions.append(view);
        if (job.status === 'queued') {
          const cancel = element('button', { className: 'secondary', type: 'button' }, 'Cancel');
          cancel.addEventListener('click', () => updateJob(job.job_id, 'cancel'));
          actions.append(cancel);
        }
        if (job.status === 'error' || job.status === 'cancelled') {
          const retry = element('button', { className: 'secondary', type: 'button' }, 'Retry');
          retry.addEventListener('click', () => updateJob(job.job_id, 'retry'));
          actions.append(retry);
        }
        row.append(label, actions);
        jobHistoryList.append(row);
      });
    }
    function refreshJobs() {
      return fetch('/api/jobs').then(async response => {
        const payload = await response.json();
        if (!response.ok) throw Error(payload.error || 'Could not load job history');
        renderJobHistory(payload.jobs || []);
      });
    }
    function updateJob(jobId, action) {
      return fetch(`/api/jobs/${encodeURIComponent(jobId)}/${action}`, { method: 'POST' })
        .then(async response => {
          const payload = await response.json();
          if (!response.ok) throw Error(payload.error || `Could not ${action} job`);
          activeJob = payload.job_id;
          result.textContent = JSON.stringify(payload, null, 2);
          return refreshJobs().then(() => action === 'retry' ? pollJob() : undefined);
        })
        .catch(error => setStatus(error.message, true));
    }
    function validateCurrent() {
      validateButton.disabled = true;
      setStatus('Validating recorder, model contracts, and runtime profiles...');
      return fetch('/api/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload()),
      }).then(async response => {
        const payload = await response.json();
        if (!response.ok) throw Error(payload.error || 'Input validation failed');
        validationReady = true;
        setStatus('Input validated. Ready to compare.');
        updateExecutionAvailability();
      }).catch(error => {
        validationReady = false;
        updateExecutionAvailability();
        setStatus(error.message, true);
      }).finally(() => { validateButton.disabled = false; });
    }
    async function pollJob() {
      if (!activeJob) return;
      const response = await fetch(`/api/jobs/${encodeURIComponent(activeJob)}`);
      const payload = await response.json();
      if (!response.ok) throw Error(payload.error || 'Could not load job');
      result.textContent = JSON.stringify(payload, null, 2);
      if (payload.status === 'queued' || payload.status === 'running') {
        window.setTimeout(() => {
          Promise.all([pollJob(), refreshServiceHealth(), refreshJobs()]).catch(error => setStatus(error.message, true));
        }, 500);
      } else {
        updateExecutionAvailability();
        setStatus(payload.status === 'complete' ? 'Comparison complete.' : (payload.error || 'Comparison failed.'), payload.status !== 'complete');
      }
    }
    validateButton.addEventListener('click', validateCurrent);
    document.getElementById('observation-path').addEventListener('input', invalidateValidation);
    function refreshServiceHealth() {
      return fetch('/api/healthz').then(async response => {
        const payload = await response.json();
        if (!response.ok) throw Error(payload.error || 'Service health check failed');
        serviceHealth = payload;
        renderServiceHealth();
      });
    }
    compareButton.addEventListener('click', async () => {
      try {
        compareButton.disabled = true;
        setStatus('Submitting comparison...');
        if (!validationReady) throw Error('Validate the input before comparing.');
        const payload = requestPayload();
        const response = await fetch('/api/compare', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        const job = await response.json();
        if (!response.ok) throw Error(job.error || 'Could not submit comparison');
        activeJob = job.job_id;
        result.textContent = JSON.stringify(job, null, 2);
        refreshJobs();
        await pollJob();
      } catch (error) {
        updateExecutionAvailability();
        setStatus(error.message, true);
      }
    });
    Promise.all([requestModels(), refreshServiceHealth(), refreshJobs()])
      .catch(error => {
        serviceHealth = { status: 'error', error: error.message };
        renderServiceHealth();
        setStatus(error.message, true);
      });
  </script>
</body>
</html>
"""

# Keep the SuperOffline page aligned with the richer comparison workspace used
# by the FastWAM branch.  The service behind these controls remains the safe
# two-to-four-model, read-only runner exposed above.
_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SuperOffline</title>
<script src="/plotly.min.js"></script>
<style>
:root{color-scheme:dark;--bg:#0d0f0c;--surface:#171916;--line:#343a32;--line-soft:#292e28;--text:#f1f4ed;--muted:#9ea79a;--green:#68dfa4;--cyan:#4dd9ff;--coral:#ff7a66;--yellow:#f3c969;--danger:#ff766f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,sans-serif}.shell{min-height:100vh}.topbar{height:72px;border-bottom:1px solid var(--line);background:#121411;display:flex;align-items:center;justify-content:space-between;padding:0 max(18px,calc((100vw - 1800px)/2));position:sticky;top:0;z-index:2}.brand{display:flex;align-items:center;gap:14px}.brand-mark{width:34px;height:34px;border:1px solid var(--green);border-radius:4px;display:grid;place-items:center;color:var(--green);font-weight:800;background:rgba(104,223,164,.08)}.brand h1{font-size:18px;line-height:1;margin:0}.brand small{display:block;color:#aab3a6;margin-top:5px;font-size:11px}.topbar-tools{display:flex;align-items:center;gap:10px}.gpu-status{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:11px}.gpu-status strong{color:#dce2d8}.app-state{display:flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:4px;background:#0d0f0c;padding:7px 10px;color:var(--muted);font-size:12px}.state-dot{width:8px;height:8px;border-radius:50%;background:#697066}.state-dot.running{background:var(--yellow)}.state-dot.complete{background:var(--green)}.state-dot.error{background:var(--danger)}main{max-width:1800px;margin:0 auto;padding:28px 18px 72px}.section{border-top:1px solid var(--line);border-bottom:1px solid var(--line);background:var(--surface);margin-bottom:24px}.section-head{min-height:54px;padding:13px 20px;border-bottom:1px solid var(--line-soft);display:flex;align-items:center;justify-content:space-between;gap:16px}.section-head h2{font-size:12px;letter-spacing:.08em;text-transform:uppercase;margin:0}.section-head .meta{color:var(--muted);font-size:12px;max-width:60%;text-align:right}.setup-grid{display:grid;grid-template-columns:minmax(460px,1.2fr) minmax(430px,1fr) minmax(340px,.82fr);min-width:1230px}.setup-column{min-width:0;padding:20px;border-right:1px solid var(--line-soft)}.setup-column:last-child{border-right:0}.column-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 16px}.column-title{margin:0;color:#dce2d8;font-size:12px;font-weight:750;letter-spacing:.06em;text-transform:uppercase}.run-column .column-title{color:var(--green)}.field{display:flex;flex-direction:column;gap:6px;margin-bottom:14px;min-width:0}.field label{font-size:11px;color:#aeb7aa}.field.wide{grid-column:1/-1}.observation-fields,.settings-grid{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(260px,.9fr);gap:0 10px}.settings-grid{grid-template-columns:1fr 1fr}.segment{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));border:1px solid #3d443a;border-radius:4px;overflow:hidden;background:#10120f}.segment button{border:0;border-right:1px solid #3d443a;border-radius:0;background:transparent;color:var(--muted);padding:8px 10px}.segment button:last-child{border-right:0}.segment button.active{background:#30382f;color:var(--green)}input,select{width:100%;min-width:0;min-height:39px;background:#10120f;color:var(--text);border:1px solid #3d443a;border-radius:4px;padding:9px 10px;outline:none}input:focus,select:focus{border-color:var(--green)}button{font:inherit;border:1px solid #495147;border-radius:4px;background:#252a24;color:var(--text);padding:9px 12px;font-weight:650;cursor:pointer}button:hover{border-color:#667061;background:#2b312a}button:disabled{opacity:.45;cursor:not-allowed}.primary{background:var(--green);border-color:var(--green);color:#07100b}.primary:hover{background:#7be7b2;border-color:#7be7b2}.quiet{color:#c1c8bd;background:transparent}.danger-button{color:#ffaaa4;background:transparent;border-color:#57403d}.icon-button{display:inline-grid;place-items:center;width:39px;height:39px;padding:0;font-size:19px;line-height:1}.observation-nav{display:grid;grid-template-columns:39px minmax(0,1fr) 39px;gap:6px}.preview-strip{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:2px}.preview-strip figure{margin:0}.preview-strip img{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#050605;border:1px solid var(--line-soft);border-radius:4px}.preview-strip figcaption{color:var(--muted);font-size:10px;margin-top:5px}.metrics{display:flex;gap:7px;flex-wrap:wrap;margin:2px 0 14px}.metric{border:1px solid var(--line);background:#10120f;color:#bdc5b9;padding:5px 8px;border-radius:3px;font-size:11px}.checkpoint-roots{display:grid;grid-template-columns:1fr 1fr;gap:0 10px}.checkpoint-list{display:flex;flex-direction:column;gap:12px;margin:4px 0 14px}.checkpoint-row{display:grid;grid-template-columns:6px 104px minmax(0,1fr) 39px;gap:8px;align-items:center;border-top:1px solid var(--line-soft);padding-top:12px}.checkpoint-row:first-child{border-top:0;padding-top:0}.checkpoint-row .color-bar{grid-column:1;grid-row:1/4;width:5px;height:100%;min-height:39px;border-radius:2px;background:var(--cyan)}.checkpoint-row:nth-child(2) .color-bar{background:var(--coral)}.checkpoint-row .backend{grid-column:2}.checkpoint-row .variant{grid-column:3}.checkpoint-row .remove{grid-column:4}.checkpoint-row .checkpoint{grid-column:2/5}.checkpoint-row .config{grid-column:2/5}.hint,.status-line{color:var(--muted);font-size:11px}.inline-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.run-area{border-top:1px solid var(--line-soft);padding-top:18px;margin-top:8px}.run-area .primary{width:100%;min-height:46px}.status-line{min-height:20px;margin-top:10px;line-height:1.45;overflow-wrap:anywhere}.job-body{display:grid;grid-template-columns:250px minmax(0,1fr);min-height:170px}.job-progress{padding:18px 20px;border-right:1px solid var(--line-soft)}.stage{font-size:18px;font-weight:700;margin-bottom:6px;overflow-wrap:anywhere}.progress-track{height:5px;background:#2a3029;margin:16px 0 9px}.progress-fill{height:100%;width:0;background:var(--green);transition:width .25s ease}.progress-copy{display:flex;justify-content:space-between;color:var(--muted);font-size:11px}.log-window{height:190px;overflow:auto;background:#090a08;padding:14px 18px;font:12px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace;color:#bac3b5}.result-body{padding:20px}.result-body pre{margin:0;max-height:520px;overflow:auto;background:#090a08;border:1px solid var(--line-soft);padding:14px;color:#bac3b5;font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace}.job-history{padding:20px}.job-row{display:flex;align-items:center;justify-content:space-between;gap:12px;border-top:1px solid var(--line-soft);padding:10px 0;color:#cbd2c7;font-size:12px}.job-actions{display:flex;gap:6px}.job-actions button{padding:5px 9px;font-size:11px}.hidden{display:none!important}@media(max-width:1280px){.setup-grid{min-width:0;grid-template-columns:repeat(2,minmax(0,1fr))}.setup-grid .run-column{grid-column:1/-1;border-top:1px solid var(--line-soft)}}@media(max-width:760px){.topbar{height:auto;min-height:72px;flex-wrap:wrap;padding:10px 12px}.topbar-tools{flex-wrap:wrap;justify-content:flex-end}main{padding:14px 10px 40px}.setup-grid{grid-template-columns:minmax(0,1fr)}.setup-column{border-right:0;border-bottom:1px solid var(--line-soft)}.setup-column:last-child{border-bottom:0}.checkpoint-roots{grid-template-columns:1fr}.checkpoint-row{grid-template-columns:6px minmax(0,1fr) 39px}.checkpoint-row .color-bar{grid-column:1;grid-row:1/4}.checkpoint-row .backend{grid-column:2}.checkpoint-row .variant{grid-column:2/4}.checkpoint-row .remove{grid-column:3}.checkpoint-row .checkpoint,.checkpoint-row .config{grid-column:2/4}.job-body{grid-template-columns:1fr}.job-progress{border-right:0;border-bottom:1px solid var(--line-soft)}.log-window{height:210px}.observation-fields,.settings-grid{grid-template-columns:1fr 1fr}}
 .results-layout{display:grid;grid-template-columns:340px minmax(0,1fr);min-height:230px}.results-sidebar{border-right:1px solid var(--line-soft);padding:20px;background:#141713}.result-observation-nav{display:grid;grid-template-columns:39px minmax(0,1fr) 39px;gap:6px}.results-control{margin-top:16px}.results-control-label{display:block;margin-bottom:6px;color:#aeb7aa;font-size:11px}.results-control .segment{grid-template-columns:repeat(3,1fr)}.toggle-button.active{border-color:var(--green);background:rgba(104,223,164,.08);color:var(--green)}.smoothness-list{display:grid;gap:8px}.smoothness-item{border-left:3px solid var(--cyan);padding:7px 0 7px 9px;color:#c8d0c4;font-size:11px}.smoothness-item:nth-child(even){border-left-color:var(--coral)}.smoothness-item strong{display:block;color:var(--text);margin-bottom:3px}.smoothness-rank{float:right;color:var(--green)}.plots-workspace{padding:20px;min-width:0}.plot-panel{border:1px solid var(--line-soft);background:#111310;min-height:190px;color:var(--muted)}.panel-title{height:42px;border-bottom:1px solid var(--line-soft);padding:0 14px;display:flex;align-items:center;justify-content:space-between;color:#dce2d8;font-size:12px}.plot{width:100%;height:440px}.projection-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}.projection-grid .plot{height:330px}.model-media{display:block;width:100%;max-height:300px;object-fit:contain;background:#050605;border:1px solid var(--line-soft);border-radius:4px}.media-label{display:block;margin:12px 0 6px;color:#aeb7aa;font-size:10px;text-transform:uppercase}.checkpoint-output-section{border-top:1px solid var(--line-soft);padding-top:20px;margin-top:20px}.checkpoint-output-section h3{margin:0 0 12px;color:#dce2d8;font-size:12px;text-transform:uppercase}.checkpoint-results{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.checkpoint-results>.warning-panel{grid-column:1/-1}.checkpoint-card{border:1px solid var(--line);border-left:4px solid var(--cyan);background:#10120f;padding:12px;min-width:0}.checkpoint-card:nth-child(2){border-left-color:var(--coral)}.checkpoint-card:nth-child(3){border-left-color:var(--yellow)}.checkpoint-card:nth-child(4){border-left-color:var(--green)}.checkpoint-card strong{display:block;color:var(--text);margin-bottom:5px}.warning-panel{border:1px solid #6e3431;background:#241312;color:#ffaaa4;padding:9px 10px;margin:0 0 10px;font-size:11px;white-space:pre-wrap}.log-line{white-space:pre-wrap;overflow-wrap:anywhere}.log-line.warning,.log-line.error{color:#ff8f87}.checkpoint-row:nth-child(3) .color-bar{background:var(--yellow)}.checkpoint-row:nth-child(4) .color-bar{background:var(--green)}@media(max-width:900px){.results-layout{grid-template-columns:1fr}.results-sidebar{border-right:0;border-bottom:1px solid var(--line-soft)}.projection-grid,.checkpoint-results{grid-template-columns:1fr}}
</style>
</head>
<body><div class="shell">
<header class="topbar"><div class="brand"><div class="brand-mark">SO</div><div><h1>SuperOffline</h1><small>Offline real robot scene checkpoint evaluation</small></div></div><div class="topbar-tools"><div class="gpu-status"><strong id="gpuMemory">GPU --</strong><span id="gpuName">checking...</span></div><button id="refreshGpu" type="button" class="icon-button" title="Refresh GPU status" aria-label="Refresh GPU status">&#8635;</button><span id="service-health" class="hint">Checking...</span><div class="app-state"><span id="stateDot" class="state-dot"></span><span id="appState">Ready</span></div></div></header>
<main>
<section class="section"><div class="section-head"><h2>Comparison Workspace</h2><span id="runSummary" class="meta">No observation run loaded</span></div><div class="setup-grid">
<div class="setup-column observation-column"><div class="column-heading"><h3 class="column-title">Observation Input</h3></div><div class="observation-fields">
<div class="field wide"><label>Observation dataset type</label><div id="observationSourceType" class="segment"><button type="button" class="active">observation_recorder</button><button type="button" disabled>LeRobot training dataset</button></div></div>
<div class="field"><label for="obsSelect">Recorder run</label><select id="obsSelect"><option value="">Use folder below</option></select></div>
<div class="field"><label for="recordSelect">Observation sample</label><div class="observation-nav"><button id="prevRecord" type="button" class="icon-button" disabled>&#8249;</button><select id="recordSelect"><option value="">Choose an input</option></select><button id="nextRecord" type="button" class="icon-button" disabled>&#8250;</button></div></div>
<div class="field wide"><label for="obs">Folder</label><input id="obs" autocomplete="off" spellcheck="false" placeholder="recorder run or observation dataset root"></div>
</div><div id="runMetrics" class="metrics"></div><div class="preview-strip"><figure><img id="preview0" alt="First camera observation"><figcaption>Camera 0</figcaption></figure><figure><img id="preview1" alt="Second camera observation"><figcaption>Camera 1</figcaption></figure></div></div>
<div class="setup-column checkpoint-column"><div class="column-heading"><h3 class="column-title">Checkpoints</h3><button id="refreshCheckpoints" type="button" class="icon-button" title="Refresh checkpoint discovery">&#8635;</button></div><div class="checkpoint-roots"><div class="field"><label for="checkpointRoot">FastWAM roots</label><input id="checkpointRoot" placeholder="one or more discovery roots"><span class="hint">Discovery only; choose a result below or enter a path.</span></div><div class="field"><label for="openpiCheckpointRoot">OpenPi root</label><input id="openpiCheckpointRoot" placeholder="checkpoint discovery root"></div></div><div id="ckpts" class="checkpoint-list"></div><div class="inline-actions"><button id="addCheckpoint" type="button" class="quiet">Add checkpoint</button><span class="hint">Compare 2 to 4 checkpoints.</span></div></div>
<div class="setup-column run-column"><div class="column-heading"><h3 class="column-title">Run Inference</h3></div><div class="settings-grid">
<div class="field wide"><label for="prompt">Prompt</label><input id="prompt" value="Place the gift into the box and cover the box with the lid"></div>
<div class="field wide"><label>Run scope</label><div id="runScope" class="segment" style="grid-template-columns:repeat(3,minmax(0,1fr))"><button type="button" data-value="all" class="active">All samples</button><button type="button" data-value="sample">Current sample</button><button type="button" data-value="stride">Sample stride</button></div></div>
<div id="sampleStrideField" class="field wide hidden"><label for="sampleStride">Observation stride</label><input id="sampleStride" type="number" min="1" step="1" value="10"></div>
<div class="field wide"><label>Output</label><div id="outputMode" class="segment"><button type="button" data-value="trajectory_only" class="active">Trajectory only</button><button type="button" data-value="trajectory_video">Trajectory + video prediction</button></div></div>
<div class="field"><label for="seed">Seed</label><input id="seed" type="number" value="42"></div><div class="field"><label for="steps">Steps</label><input id="steps" type="number" min="1" value="10"></div><div class="field"><label for="device">Device</label><select id="device"><option>cuda</option><option>cpu</option></select></div>
</div><div class="run-area"><div id="validationWarnings" class="warning-panel hidden"></div><button id="validate" type="button" class="quiet" style="width:100%;margin-bottom:8px">Validate input</button><button id="run" type="button" class="primary" disabled>Run Inference</button><button id="compare" type="button" class="hidden">Compare</button><div id="status" class="status-line">Validate the observation and checkpoint contracts before running.</div></div></div>
</div></section>
<section id="jobSection" class="section"><div class="section-head"><h2>Job Log</h2><span id="jobId" class="meta">Idle</span></div><div class="job-body"><div class="job-progress"><div id="jobStage" class="stage">Ready</div><div id="jobDetail" class="hint">Waiting for a comparison job</div><button id="cancelJob" type="button" class="danger-button" disabled>Cancel run</button><div class="progress-track"><div id="progressFill" class="progress-fill"></div></div><div class="progress-copy"><span id="progressPercent">0%</span><span id="progressObservation">-</span></div></div><div id="logWindow" class="log-window" role="log" aria-live="polite">Runtime stages and debug messages will appear here.</div></div></section>
<section class="section job-history"><div class="section-head"><h2>Recent Jobs</h2><span class="meta">Local runtime history</span></div><div id="job-history-list">No jobs yet.</div></section>
<section id="resultControls" class="section result-body"><div class="section-head" style="margin:-20px -20px 20px"><h2>Comparison Results</h2><span id="resultJob" class="meta"></span></div><div class="results-layout"><aside class="results-sidebar"><h3 class="column-title">Observation</h3><div class="result-observation-nav"><button id="prevResult" type="button" class="icon-button" disabled>&#8249;</button><select id="resultNumber"><option value="">No result yet</option></select><button id="nextResult" type="button" class="icon-button" disabled>&#8250;</button></div><div class="preview-strip" style="margin-top:12px"><figure><img id="resultPreview0" alt="Left camera result observation"><figcaption>Left camera</figcaption></figure><figure><img id="resultPreview1" alt="Right camera result observation"><figcaption>Right camera</figcaption></figure></div><div class="results-control"><span class="results-control-label">2D projection</span><div id="projectionMode" class="segment"><button type="button" data-projection="xy" class="active">XY</button><button type="button" data-projection="xz">XZ</button><button type="button" data-projection="yz">YZ</button></div></div><div class="results-control"><span class="results-control-label">3D arms</span><div id="trajectoryArm" class="segment"><button type="button" data-arm="both" class="active">Both</button><button type="button" data-arm="left">Left</button><button type="button" data-arm="right">Right</button></div></div><div class="results-control"><button id="zeroOrigin" type="button" class="quiet toggle-button" style="width:100%" aria-pressed="false">Zero initial point</button><button id="toggleVideos" type="button" class="quiet" style="width:100%;margin-top:8px" disabled>Play all videos</button></div><div class="results-control"><span class="results-control-label">Chunk Smoothness <span id="smoothnessScope" class="hint">Pending</span></span><div id="smoothnessMetrics" class="smoothness-list"><span class="hint">Available after inference.</span></div></div><div class="hint" style="margin-top:16px">Read-only result view. No device or model server is started by this page.</div></aside><div id="plots" class="plots-workspace hidden"><article class="plot-panel"><div class="panel-title"><span>3D End-Effector Trajectory</span><span id="trajectoryReference" class="hint">absolute coordinates / both arms</span></div><div id="plot3d" class="plot"></div></article><div class="projection-grid"><article class="plot-panel"><div class="panel-title"><span>Left Arm</span><span id="leftProjection" class="hint">XY projection</span></div><div id="plotLeft" class="plot"></div></article><article class="plot-panel"><div class="panel-title"><span>Right Arm</span><span id="rightProjection" class="hint">XY projection</span></div><div id="plotRight" class="plot"></div></article></div></div></div><div class="checkpoint-output-section"><h3>Checkpoint Outputs</h3><div id="checkpointResults" class="checkpoint-results"><pre id="result">Waiting for a comparison.</pre></div></div></section>
</main></div>
<script>
const state=[{backend:'fastwam',variantId:'fastwam',checkpoint:''},{backend:'fastwam',variantId:'fastwam',checkpoint:''}];let catalog=[];let executionInfo={};let activeJob=null;let validationReady=false;let serviceHealth={};let discoveredObservations=[];let discoveredCheckpoints=[];let selectedSamples=[];let outputMode='trajectory_only';
const $=id=>document.getElementById(id);const status=$('status');const runButton=$('run');const result=$('result');let trajectoryData=null;let smoothnessData=null;let trajectoryProjection='xy';let trajectoryArm='both';let trajectoryZeroed=false;let activeResultPayload=null;let activeResultJob=null;let videosPlaying=false;
function setStatus(message,error=false){status.textContent=message;status.style.color=error?'var(--danger)':'var(--muted)';$('stateDot').className='state-dot '+(error?'error':'')}
function adapterById(id){return catalog.find(item=>item.adapter_id===id)}
function adapterForModel(model){return adapterById(model.backend==='openpi'?'openpi':'fastwam')}
function unavailableSelectedAdapter(){return state.map(adapterForModel).find(adapter=>adapter&&adapter.available===false)||null}
function renderValidationWarnings(warnings){const root=$('validationWarnings');const messages=(Array.isArray(warnings)?warnings:[]).map(item=>item?.message).filter(Boolean);root.textContent=messages.join('\n');root.classList.toggle('hidden',!messages.length)}
function invalidateValidation(){validationReady=false;renderValidationWarnings([]);updateRunAvailability()}
function selectedContract(index){const model=state[index],adapter=adapterForModel(model);if(!adapter)return null;if(model.backend==='openpi')return (adapter.config_contracts||[]).find(item=>item.config_name===adapter.default_config_name)||null;const variant=model.backend==='fasterwam'?'fasterwam':(model.variantId||'fastwam');return (adapter.variants||[]).find(item=>item.id===variant)||null}
function compatible(){const contracts=state.map((_,index)=>selectedContract(index));const first=contracts[0];return Boolean(first&&contracts.every(item=>item&&item.action_frame_id===first.action_frame_id&&item.action_units===first.action_units))}
function option(select,value,text,selected){const node=document.createElement('option');node.value=value;node.textContent=text;node.selected=Boolean(selected);select.append(node)}
function field(parent,labelText,control){const label=document.createElement('label');label.textContent=labelText;label.append(control);parent.append(label)}
function discoveredForBackend(backend){return discoveredCheckpoints.filter(item=>backend==='openpi'?item.backend==='openpi':backend==='fasterwam'?item.backend==='fasterwam':item.backend==='fastwam')}
function renderCheckpoints(){const root=$('ckpts');root.replaceChildren();state.forEach((model,index)=>{const row=document.createElement('div');row.className='checkpoint-row';const bar=document.createElement('span');bar.className='color-bar';row.append(bar);const backend=document.createElement('select');backend.className='backend';backend.setAttribute('aria-label','Checkpoint '+(index+1)+' backend');[['fastwam','FastWAM (auto)'],['fasterwam','FasterWAM (C4)'],['openpi','OpenPi']].forEach(item=>option(backend,item[0],item[1],item[0]===model.backend));backend.addEventListener('change',()=>{model.backend=backend.value;model.variantId=backend.value==='fasterwam'?'fasterwam':backend.value==='fastwam'?'fastwam':'';invalidateValidation();renderCheckpoints()});row.append(backend);const discovered=document.createElement('select');discovered.className='variant';option(discovered,'','Choose discovered checkpoint...',true);discoveredForBackend(model.backend).forEach(item=>option(discovered,item.path,item.display_name||item.name,item.path===model.checkpoint));discovered.addEventListener('change',()=>{if(!discovered.value)return;const item=discoveredCheckpoints.find(entry=>entry.path===discovered.value);model.checkpoint=discovered.value;if(item?.variant_id)model.variantId=item.variant_id;invalidateValidation();renderCheckpoints()});row.append(discovered);const remove=document.createElement('button');remove.className='remove danger-button icon-button';remove.type='button';remove.title='Remove checkpoint';remove.setAttribute('aria-label','Remove checkpoint '+(index+1));remove.textContent='×';remove.disabled=state.length<=2;remove.addEventListener('click',()=>{if(state.length<=2)return;state.splice(index,1);invalidateValidation();renderCheckpoints()});row.append(remove);const checkpoint=document.createElement('input');checkpoint.className='checkpoint';checkpoint.value=model.checkpoint;checkpoint.placeholder='or enter checkpoint path';checkpoint.addEventListener('input',()=>{model.checkpoint=checkpoint.value;const item=discoveredCheckpoints.find(entry=>entry.path===model.checkpoint);if(item?.variant_id)model.variantId=item.variant_id;invalidateValidation()});row.append(checkpoint);root.append(row)});$('addCheckpoint').disabled=state.length>=4;updateRunAvailability()}
function renderObservationPreviews(){const path=$('obs').value.trim();const index=Number($('recordSelect').value);for(let camera=0;camera<2;camera++){const image=$('preview'+camera);if(!path||!Number.isInteger(index)){image.removeAttribute('src');continue}image.src='/api/observation/preview?path='+encodeURIComponent(path)+'&inference_index='+index+'&camera_index='+camera+'&v='+Date.now();image.onerror=()=>image.removeAttribute('src')}}
function renderObservationSamples(){const select=$('recordSelect');select.replaceChildren();if(!selectedSamples.length){option(select,'','Choose an observation sample',true);$('runMetrics').replaceChildren();$('runSummary').textContent='No observation run loaded';renderObservationPreviews();return}selectedSamples.forEach((sample,index)=>option(select,String(index),`#${index+1} · ${sample.timestamp_ns||'timestamp unavailable'}`,index===0));select.value='0';const sample=selectedSamples[0];const metrics=$('runMetrics');metrics.replaceChildren();[['Samples',selectedSamples.length],['Timestamp',sample.timestamp_ns||'n/a'],['Artifact',sample.file]].forEach(([label,value])=>{const node=document.createElement('span');node.className='metric';node.textContent=`${label}: ${value}`;metrics.append(node)});$('runSummary').textContent=`${selectedSamples.length} recorded observation samples`;$('prevRecord').disabled=true;$('nextRecord').disabled=selectedSamples.length<2;renderObservationPreviews()}
async function loadSamples(path){if(!path){selectedSamples=[];renderObservationSamples();return}const response=await fetch('/api/observations/samples?path='+encodeURIComponent(path));const payload=await response.json();if(!response.ok)throw Error(payload.error||'Could not load observation samples');selectedSamples=Array.isArray(payload)?payload:[];renderObservationSamples()}
async function loadObservations(){const response=await fetch('/api/observations');const payload=await response.json();if(!response.ok)throw Error(payload.error||'Could not discover observation runs');discoveredObservations=Array.isArray(payload)?payload:[];const select=$('obsSelect');select.replaceChildren();option(select,'','Choose a recorder run',true);discoveredObservations.forEach(item=>option(select,item.path,item.path));if(discoveredObservations.length){select.value=discoveredObservations[0].path;$('obs').value=discoveredObservations[0].path;await loadSamples(discoveredObservations[0].path)}}
function requestModel(model){if(model.backend==='openpi'){const adapter=adapterById('openpi');return{adapter_id:'openpi',checkpoint:model.checkpoint.trim(),config_name:adapter?.default_config_name||'policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi'}}const payload={adapter_id:'fastwam',checkpoint:model.checkpoint.trim()};if(model.backend==='fasterwam')payload.variant_id='fasterwam';else if(model.variantId&&model.variantId!=='fastwam')payload.variant_id=model.variantId;return payload}
function requestPayload(){const payload={observation_path:$('obs').value.trim(),models:state.map(requestModel)};const prompt=$('prompt').value.trim();if(prompt)payload.metadata={prompt,run_scope:document.querySelector('#runScope .active')?.dataset.value||'all',sample_stride:Number($('sampleStride').value)||10,output_mode:outputMode,inference_index:Number($('recordSelect').value)||0,seed:Number($('seed').value)||42,num_inference_steps:Number($('steps').value)||10,device:$('device').value};return payload}
function updateRunAvailability(){const unavailable=unavailableSelectedAdapter();const ready=!unavailable&&compatible()&&validationReady;runButton.disabled=!ready;if(unavailable)setStatus(unavailable.availability_error||'Selected model is unavailable.',true);else if(!compatible())setStatus('Choose two compatible checkpoint contracts.',true);else if(!validationReady)setStatus('Validate the observation and checkpoint contracts before running.');else setStatus('Ready for a read-only comparison.')}
function renderServiceHealth(){const scheduler=serviceHealth.scheduler||{};const label=serviceHealth.status==='ok'?`Ready · ${scheduler.running_jobs||0} running · ${scheduler.queued_jobs||0} queued`:(serviceHealth.error||'Unavailable');$('appState').textContent=label;$('service-health').textContent=label}
async function loadCatalog(){const response=await fetch('/api/models');const payload=await response.json();if(!response.ok)throw Error(payload.error||'Could not load model catalog');catalog=payload.models||[];renderCheckpoints()}
async function loadCheckpointCatalog(){const query=new URLSearchParams({fastwam_dir:$('checkpointRoot').value.trim(),openpi_dir:$('openpiCheckpointRoot').value.trim()});const response=await fetch('/api/checkpoints?'+query);const payload=await response.json();if(!response.ok)throw Error(payload.error||'Could not discover checkpoints');discoveredCheckpoints=Array.isArray(payload)?payload:[];renderCheckpoints();setStatus(`Found ${discoveredCheckpoints.length} local checkpoint${discoveredCheckpoints.length===1?'':'s'}.`)}
async function refreshHealth(){const response=await fetch('/api/healthz');serviceHealth=await response.json();renderServiceHealth()}
async function refreshGpu(){const response=await fetch('/api/gpu');const payload=await response.json();const gpu=(payload.gpus||[])[0];$('gpuMemory').textContent=gpu?`${gpu.memory_used_mib} / ${gpu.memory_total_mib} MiB`:'GPU --';$('gpuName').textContent=gpu?gpu.name:(payload.error||'unavailable')}
function renderJobs(jobs){const root=$('job-history-list');root.replaceChildren();if(!jobs.length){root.textContent='No jobs yet.';return}jobs.forEach(job=>{const row=document.createElement('div');row.className='job-row';const label=document.createElement('span');label.textContent=`${job.status} · ${job.job_id.slice(0,8)} · ${job.stage||'queued'} · ${Number(job.progress||0)}%`;const actions=document.createElement('span');actions.className='job-actions';const view=document.createElement('button');view.textContent='View';view.onclick=()=>{activeJob=job.job_id;renderResult(job);$('jobId').textContent=job.job_id;$('jobStage').textContent=job.stage||job.status};actions.append(view);if(job.status==='queued'||job.status==='running'){const cancel=document.createElement('button');cancel.textContent='Cancel';cancel.onclick=()=>jobAction(job.job_id,'cancel');actions.append(cancel)}if(job.status==='error'||job.status==='cancelled'){const retry=document.createElement('button');retry.textContent='Retry';retry.onclick=()=>jobAction(job.job_id,'retry');actions.append(retry)}row.append(label,actions);root.append(row)})}
function renderJobLog(payload){const root=$('logWindow');const logs=Array.isArray(payload.logs)?payload.logs:[];root.replaceChildren();if(!logs.length){root.textContent='Runtime stages and debug messages will appear here.';return}logs.forEach(item=>{const line=document.createElement('div');line.className='log-line '+(item.severity||'info');const stamp=String(item.timestamp||'').replace('T',' ').replace('+00:00','Z');line.textContent=`${stamp}  ${String(item.stage||'runtime').padEnd(20)} ${item.message||''}`;root.append(line)});root.scrollTop=root.scrollHeight}
function renderSmoothness(){
  const root=$('smoothnessMetrics');root.replaceChildren();const checkpoints=smoothnessData?.checkpoints||[];$('smoothnessScope').textContent=smoothnessData?`${smoothnessData.observation_count||0} chunks`:'Pending';
  if(!checkpoints.length){const empty=document.createElement('span');empty.className='hint';empty.textContent='Available after inference.';root.append(empty);return}
  checkpoints.forEach(checkpoint=>{const metrics=checkpoint.metrics;const node=document.createElement('div');node.className='smoothness-item';node.style.borderLeftColor=checkpoint.color||'#4dd9ff';const title=document.createElement('strong');title.textContent=checkpoint.checkpoint_name||'checkpoint';if(checkpoint.rank){const rank=document.createElement('span');rank.className='smoothness-rank';rank.textContent=`#${checkpoint.rank}`;title.append(rank)}node.append(title);const detail=document.createElement('span');detail.textContent=metrics?`roughness ${Number(metrics.roughness).toPrecision(3)} · speed P50 ${Number(metrics.speed_median_mm_per_step).toPrecision(3)} mm/step · accel P95 ${Number(metrics.acceleration_p95_mm_per_step2).toPrecision(3)} · jerk P95 ${Number(metrics.jerk_p95_mm_per_step3).toPrecision(3)}`:'No valid trajectory data';node.append(detail);root.append(node)})
}
function trajectoryPoints(item,arm){const source=(item[arm]||[]).map(point=>point.map(Number));if(!trajectoryZeroed||!source.length)return source;const origin=source[0];return source.map(point=>point.map((value,index)=>value-origin[index]))}
function renderTrajectory(){
  if(!window.Plotly)return;const trajectories=trajectoryData?.trajectories||[];if(!trajectories.length){for(const id of ['plot3d','plotLeft','plotRight'])Plotly.purge(id);return}
  const colors=['#4dd9ff','#ff7a66','#72e08a','#f3c969'];const selectedArms=trajectoryArm==='both'?['left','right']:[trajectoryArm];const traces3d=[];
  trajectories.forEach((item,index)=>selectedArms.forEach((arm,armIndex)=>{const points=trajectoryPoints(item,arm);if(!points.length)return;const color=colors[(index*2+armIndex)%colors.length];traces3d.push({type:'scatter3d',mode:'lines+markers',name:`${item.label} / ${arm}`,x:points.map(point=>point[0]),y:points.map(point=>point[1]),z:points.map(point=>point[2]),line:{color,width:5},marker:{color,size:2,symbol:arm==='left'?'circle':'diamond'}})}));
  const transparent={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'#111310',font:{color:'#cbd2c7'},margin:{l:45,r:16,t:16,b:42},legend:{orientation:'h'}};Plotly.react('plot3d',traces3d,{...transparent,margin:{l:0,r:0,t:10,b:0},scene:{bgcolor:'#111310',aspectmode:'data',xaxis:{title:trajectoryZeroed?'x offset (m)':'x (m)',gridcolor:'#343a32'},yaxis:{title:trajectoryZeroed?'y offset (m)':'y (m)',gridcolor:'#343a32'},zaxis:{title:trajectoryZeroed?'z offset (m)':'z (m)',gridcolor:'#343a32'}}},{responsive:true,displaylogo:false,scrollZoom:true});
  const axis=trajectoryProjection==='xy'?[0,1]:trajectoryProjection==='xz'?[0,2]:[1,2];const axisNames=trajectoryProjection.split('');for(const arm of ['left','right']){const traces=trajectories.map((item,index)=>{const points=trajectoryPoints(item,arm);return {type:'scatter',mode:'lines+markers',name:item.label,x:points.map(point=>point[axis[0]]),y:points.map(point=>point[axis[1]]),line:{color:colors[index%colors.length],width:3},marker:{color:colors[index%colors.length],size:4,symbol:arm==='left'?'circle':'diamond'}}});Plotly.react(arm==='left'?'plotLeft':'plotRight',traces,{...transparent,xaxis:{title:`${axisNames[0]}${trajectoryZeroed?' offset':''} (m)`,gridcolor:'#343a32'},yaxis:{title:`${axisNames[1]}${trajectoryZeroed?' offset':''} (m)`,gridcolor:'#343a32'},showlegend:arm==='left'},{responsive:true,displaylogo:false,scrollZoom:true})}
  $('trajectoryReference').textContent=`${trajectoryZeroed?'zeroed offsets':'absolute coordinates'} / ${trajectoryArm} arms`;$('leftProjection').textContent=`${trajectoryProjection.toUpperCase()} projection`;$('rightProjection').textContent=`${trajectoryProjection.toUpperCase()} projection`;
}
async function loadTrajectory(jobId,sampleIndex=0){try{const response=await fetch('/api/jobs/'+encodeURIComponent(jobId)+'/trajectory?sample='+sampleIndex);const payload=await response.json();if(!response.ok)throw Error(payload.error||'Trajectory is unavailable');trajectoryData=payload;renderTrajectory()}catch(error){trajectoryData=null;renderTrajectory()}}
async function loadSmoothness(jobId){try{const response=await fetch('/api/jobs/'+encodeURIComponent(jobId)+'/smoothness');const payload=await response.json();if(!response.ok)throw Error(payload.error||'Smoothness is unavailable');if(activeResultJob!==jobId)return;smoothnessData=payload;renderSmoothness()}catch(error){smoothnessData=null;$('smoothnessScope').textContent='Unavailable';renderSmoothness()}}
function artifactUrl(jobId,parts){return '/api/jobs/'+encodeURIComponent(jobId)+'/artifact/'+parts.map(part=>encodeURIComponent(String(part))).join('/')}
function appendArtifactLabel(card,text){const label=document.createElement('span');label.className='media-label';label.textContent=text;card.append(label)}
function renderResult(payload,sampleIndex=null){
  renderJobLog(payload);
  const jobChanged=activeResultJob!==payload.job_id;
  activeResultJob=payload.job_id||null;activeResultPayload=payload;
  if(jobChanged){smoothnessData=null;renderSmoothness()}
  const rootData=payload.result||payload;
  const samples=Array.isArray(rootData.samples)?rootData.samples:[];
  const select=$('resultNumber');
  if(samples.length&&(jobChanged||select.options.length!==samples.length||select.options[0]?.value!=="0")){
    select.replaceChildren();samples.forEach((sample,index)=>option(select,String(index),`#${Number(sample.inference_index)+1} · ${sample.observation_timestamp_ns||'timestamp unavailable'}`,index===0));
  }else if(!samples.length&&jobChanged){select.replaceChildren();option(select,'','No result yet',true)}
  const requestedIndex=sampleIndex===null?Number(select.value)||0:sampleIndex;
  const selectedIndex=Math.max(0,Math.min(requestedIndex,Math.max(0,samples.length-1)));
  if(samples.length)select.value=String(selectedIndex);
  const data=samples[selectedIndex]||rootData;
  const models=Array.isArray(data.models)?data.models:[];
  $('prevResult').disabled=!samples.length||selectedIndex<=0;
  $('nextResult').disabled=!samples.length||selectedIndex>=samples.length-1;
  $('resultJob').textContent=payload.job_id?`Job ${payload.job_id.slice(0,8)} · sample ${selectedIndex+1}/${samples.length||1}`:'';
  result.textContent=JSON.stringify(data,null,2);
  const observationIndex=Number(data.inference_index??selectedIndex);
  if(samples.length){for(let camera=0;camera<2;camera++)$('resultPreview'+camera).src='/api/observation/preview?path='+encodeURIComponent(rootData.observation_snapshot||$('obs').value.trim())+'&inference_index='+observationIndex+'&camera_index='+camera}
  const output=$('checkpointResults');output.replaceChildren();
  const warnings=Array.isArray(payload.warnings)?payload.warnings:[];
  warnings.forEach(item=>{const warning=document.createElement('div');warning.className='warning-panel';warning.textContent=item.message||'Task warning';output.append(warning)});
  models.forEach((entry,index)=>{
    const model=entry.model||{};const summary=entry.summary||{};const artifacts=summary.artifacts||{};
    const card=document.createElement('article');card.className='checkpoint-card';
    const title=document.createElement('strong');title.textContent=`Checkpoint ${index+1}: ${model.label||model.config_name||'registered model'}`;card.append(title);
    const shape=Array.isArray(summary.action_shape)?summary.action_shape.join(' × '):'unknown';
    const detail=document.createElement('div');const seconds=Number(summary.inference_seconds);detail.textContent=`Action ${shape}${Number.isFinite(seconds)?` · ${seconds.toFixed(2)} s`:''} · ${summary.source_profile||'source contract verified'}`;card.append(detail);
    const taskValidation=entry.task_validation||{};
    if(taskValidation.status==='mismatch_warning'){const warning=document.createElement('div');warning.className='warning-panel';warning.style.marginTop='10px';warning.textContent=`Task mismatch: trained for “${taskValidation.training_task}”; running “${taskValidation.requested_task}”.`;card.append(warning)}
    if(payload.job_id&&entry.output_dir&&typeof artifacts.input_preview==='string'){
      appendArtifactLabel(card,'Model input');const image=document.createElement('img');image.className='model-media';image.alt=`Model input for ${model.label||'checkpoint'}`;image.src=artifactUrl(payload.job_id,[entry.output_dir,artifacts.input_preview]);card.append(image);
    }
    if(payload.job_id&&entry.output_dir&&typeof artifacts.video_prediction==='string'){
      appendArtifactLabel(card,'Predicted video');const video=document.createElement('video');video.className='model-media prediction-video';video.controls=true;video.muted=true;video.playsInline=true;video.preload='metadata';video.src=artifactUrl(payload.job_id,[entry.output_dir,artifacts.video_prediction]);card.append(video);
    }else if((payload.request?.metadata||{}).output_mode==='trajectory_video'){
      const empty=document.createElement('div');empty.className='hint';empty.style.marginTop='10px';empty.textContent=model.adapter_id==='fastwam'?'No decoded video was produced.':'This backend is action-only; no decoded video is expected.';card.append(empty);
    }
    output.append(card);
  });
  if(!models.length){output.append(result)}
  videosPlaying=false;$('toggleVideos').textContent='Play all videos';$('toggleVideos').disabled=!output.querySelector('.prediction-video');
  const hasCompletedResult=payload.status==='complete'&&models.length>0;$('plots').classList.toggle('hidden',!hasCompletedResult);
  if(!hasCompletedResult){trajectoryData=null;renderTrajectory();return}
  loadTrajectory(payload.job_id,selectedIndex);if(jobChanged||!smoothnessData)loadSmoothness(payload.job_id);
}
async function refreshJobs(){const response=await fetch('/api/jobs');const payload=await response.json();if(!response.ok)throw Error(payload.error||'Could not load jobs');renderJobs(payload.jobs||[])}
async function jobAction(jobId,action){try{const response=await fetch('/api/jobs/'+encodeURIComponent(jobId)+'/'+action,{method:'POST'});const payload=await response.json();if(!response.ok)throw Error(payload.error||'Could not update job');activeJob=payload.job_id;renderResult(payload);await refreshJobs();if(action==='retry')pollJob()}catch(error){setStatus(error.message,true)}}
async function pollJob(){if(!activeJob)return;const response=await fetch('/api/jobs/'+encodeURIComponent(activeJob));const payload=await response.json();if(!response.ok)throw Error(payload.error||'Could not load job');renderResult(payload);const progress=Math.max(0,Math.min(100,Number(payload.progress||0)));$('jobId').textContent=payload.job_id;$('jobStage').textContent=payload.stage||payload.status;$('jobDetail').textContent=payload.error||payload.latest_checkpoint||'Runtime profile comparison';$('progressFill').style.width=`${progress}%`;$('progressPercent').textContent=`${progress}%`;$('progressObservation').textContent=payload.latest_observation?`Observation ${payload.latest_observation}`:(payload.status==='complete'?`${(payload.result||{}).observation_count||0} chunks`:'-');$('cancelJob').disabled=!['queued','running'].includes(payload.status);$('stateDot').className='state-dot '+(payload.status==='complete'?'complete':payload.status==='error'||payload.status==='cancelled'?'error':['running','cancelling'].includes(payload.status)?'running':'');if(['queued','running','cancelling'].includes(payload.status)){setTimeout(()=>Promise.all([pollJob(),refreshJobs(),refreshHealth()]).catch(error=>setStatus(error.message,true)),500)}else{setStatus(payload.status==='complete'?'Comparison complete.':payload.status==='cancelled'?'Comparison cancelled.':(payload.error||'Comparison failed.'),payload.status==='error');validationReady=false;updateRunAvailability();await refreshJobs()}}
async function validateCurrent(){try{$('validate').disabled=true;renderValidationWarnings([]);setStatus('Validating recorder, model contracts, and runtime profiles...');const response=await fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(requestPayload())});const payload=await response.json();if(!response.ok)throw Error(payload.error||'Input validation failed');validationReady=true;renderValidationWarnings(payload.warnings||[]);setStatus((payload.warnings||[]).length?'Input validated with task warnings. Inference may continue.':'Input validated. Ready to run inference.');updateRunAvailability()}catch(error){validationReady=false;renderValidationWarnings([]);updateRunAvailability();setStatus(error.message,true)}finally{$('validate').disabled=false}}
$('validate').addEventListener('click',validateCurrent);$('obs').addEventListener('input',()=>{invalidateValidation();$('obsSelect').value='';loadSamples($('obs').value.trim()).catch(error=>setStatus(error.message,true))});$('obsSelect').addEventListener('change',()=>{const path=$('obsSelect').value;$('obs').value=path;invalidateValidation();loadSamples(path).catch(error=>setStatus(error.message,true))});$('recordSelect').addEventListener('change',()=>{const index=Number($('recordSelect').value);const sample=selectedSamples[index];if(!sample)return;const metrics=$('runMetrics');metrics.replaceChildren();[['Sample',index+1],['Timestamp',sample.timestamp_ns||'n/a'],['Artifact',sample.file]].forEach(([label,value])=>{const node=document.createElement('span');node.className='metric';node.textContent=`${label}: ${value}`;metrics.append(node)});$('prevRecord').disabled=index<=0;$('nextRecord').disabled=index>=selectedSamples.length-1;renderObservationPreviews()});$('prevRecord').addEventListener('click',()=>{$('recordSelect').value=String(Math.max(0,Number($('recordSelect').value)-1));$('recordSelect').dispatchEvent(new Event('change'))});$('nextRecord').addEventListener('click',()=>{$('recordSelect').value=String(Math.min(selectedSamples.length-1,Number($('recordSelect').value)+1));$('recordSelect').dispatchEvent(new Event('change'))});$('checkpointRoot').addEventListener('change',()=>{invalidateValidation();loadCheckpointCatalog().catch(error=>setStatus(error.message,true))});$('openpiCheckpointRoot').addEventListener('change',()=>{invalidateValidation();loadCheckpointCatalog().catch(error=>setStatus(error.message,true))});$('refreshCheckpoints').addEventListener('click',()=>{invalidateValidation();loadCheckpointCatalog().catch(error=>setStatus(error.message,true))});$('addCheckpoint').addEventListener('click',()=>{if(state.length>=4)return;state.push({backend:'fastwam',variantId:'fastwam',checkpoint:''});invalidateValidation();renderCheckpoints()});$('runScope').addEventListener('click',event=>{const button=event.target.closest('button[data-value]');if(!button)return;document.querySelectorAll('#runScope button').forEach(item=>item.classList.toggle('active',item===button));$('sampleStrideField').classList.toggle('hidden',button.dataset.value!=='stride');invalidateValidation()});$('sampleStride').addEventListener('input',invalidateValidation);$('cancelJob').addEventListener('click',()=>activeJob&&jobAction(activeJob,'cancel'));$('run').addEventListener('click',async()=>{try{runButton.disabled=true;setStatus('Submitting comparison...');const response=await fetch('/api/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(requestPayload())});const payload=await response.json();if(!response.ok)throw Error(payload.error||'Could not submit comparison');activeJob=payload.job_id;renderResult(payload);await refreshJobs();await pollJob()}catch(error){setStatus(error.message,true);updateRunAvailability()}});
$('projectionMode').addEventListener('click',event=>{const button=event.target.closest('button[data-projection]');if(!button)return;trajectoryProjection=button.dataset.projection;document.querySelectorAll('#projectionMode button').forEach(item=>item.classList.toggle('active',item===button));renderTrajectory()});
$('trajectoryArm').addEventListener('click',event=>{const button=event.target.closest('button[data-arm]');if(!button)return;trajectoryArm=button.dataset.arm;document.querySelectorAll('#trajectoryArm button').forEach(item=>item.classList.toggle('active',item===button));renderTrajectory()});
$('zeroOrigin').addEventListener('click',()=>{trajectoryZeroed=!trajectoryZeroed;$('zeroOrigin').classList.toggle('active',trajectoryZeroed);$('zeroOrigin').setAttribute('aria-pressed',String(trajectoryZeroed));renderTrajectory()});
$('refreshGpu').addEventListener('click',()=>refreshGpu().catch(error=>{$('gpuName').textContent=error.message}));
$('outputMode').addEventListener('click',event=>{const button=event.target.closest('button[data-value]');if(!button)return;outputMode=button.dataset.value;document.querySelectorAll('#outputMode button').forEach(item=>item.classList.toggle('active',item===button));invalidateValidation()});
$('toggleVideos').addEventListener('click',()=>{const videos=[...document.querySelectorAll('.prediction-video')];if(!videos.length)return;videosPlaying=!videosPlaying;$('toggleVideos').textContent=videosPlaying?'Pause all videos':'Play all videos';videos.forEach(video=>{if(videosPlaying){video.currentTime=0;video.play().catch(()=>{})}else video.pause()})});
$('resultNumber').addEventListener('change',()=>activeResultPayload&&renderResult(activeResultPayload,Number($('resultNumber').value)));$('prevResult').addEventListener('click',()=>{if(!activeResultPayload)return;const index=Math.max(0,Number($('resultNumber').value)-1);renderResult(activeResultPayload,index)});$('nextResult').addEventListener('click',()=>{if(!activeResultPayload)return;const samples=(activeResultPayload.result||activeResultPayload).samples||[];const index=Math.min(samples.length-1,Number($('resultNumber').value)+1);renderResult(activeResultPayload,index)});
Promise.all([loadCatalog(),loadObservations(),loadCheckpointCatalog(),refreshHealth(),refreshGpu(),refreshJobs()]).catch(error=>{setStatus(error.message,true);$('stateDot').className='state-dot error'});
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
