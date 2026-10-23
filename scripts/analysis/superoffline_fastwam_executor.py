"""Built-in, read-only SuperOffline executor for registered real-robot models.

This module orchestrates the existing offline evaluator in a verified runtime
profile.  It never imports a model policy into the browser process, and it
never opens control shared memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from utils.openpi_checkpoint import (
    OpenPiCheckpointContractError,
    discover_openpi_norm_stats_assets,
)
from utils.superoffline_action_calibration import (
    ActionCalibrationBinding,
    ActionCalibrationContract,
    ActionCalibrationError,
    snapshot_action_calibrations,
    validate_action_calibration_config,
)
from utils.superoffline_observation import (
    ObservationBinding,
    ObservationContractError,
    write_observation_bindings,
)
from utils.superoffline_preflight import (
    PreparedSuperOfflineInput,
    SuperOfflinePreflightError,
    prepare_superoffline_input,
)
from utils.superoffline_registry import (
    DUAL_ARM_TCP_ACTION_CONTRACT,
    DUAL_ARM_TCP_ACTION_FRAME,
    DUAL_ARM_TCP_ACTION_LAYOUT,
    DUAL_ARM_TCP_ACTION_UNITS,
    SuperOfflineAdapterError,
    SuperOfflineComparisonRequest,
    SuperOfflineExecutionError,
    SuperOfflineRuntimeContext,
)

_RECORDER_SCHEMA = "superinference.timestamped_observation"
_JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_RESULT_CACHE_VERSION = 1


def _safe_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise SuperOfflineExecutionError(f"metadata.{field} must be a non-empty string")
    return value.strip()


def _metadata_integer(
    metadata: Mapping[str, Any],
    field: str,
    *,
    default: int,
    minimum: int | None = None,
) -> int:
    raw_value = metadata.get(field, default)
    if isinstance(raw_value, bool):
        raise SuperOfflineExecutionError(f"metadata.{field} must be an integer")
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise SuperOfflineExecutionError(f"metadata.{field} must be an integer") from exc
    if minimum is not None and value < minimum:
        raise SuperOfflineExecutionError(
            f"metadata.{field} must be at least {minimum}"
        )
    return value


def _nested_text(payload: Mapping[str, Any], *path: str) -> str | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _normalized_task(value: str) -> str:
    return " ".join(value.split()).casefold()


class RealRobotOfflineComparisonExecutor:
    """Run two to four registered real-robot models through runtime profiles."""

    def __init__(
        self,
        *,
        repo_root: Path,
        command_runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self._repo_root = Path(repo_root).resolve()
        self._command_runner = command_runner
        self._runtime_profiles = (
            self._repo_root / "scripts" / "diagnostics" / "runtime_profiles.py"
        )
        self._evaluators = {
            "fastwam": (
                self._repo_root
                / "scripts"
                / "analysis"
                / "evaluate_fastwam_reverse_offline.py"
            ),
            "openpi": (
                self._repo_root
                / "scripts"
                / "analysis"
                / "evaluate_openpi_offline.py"
            ),
            "lingbotva": (
                self._repo_root
                / "scripts"
                / "analysis"
                / "evaluate_lingbotva_offline.py"
            ),
        }
        self._output_root = self._repo_root / "logs" / "superoffline"

    def public_metadata(self) -> dict[str, Any]:
        return {
            "mode": "builtin-real-robot-offline",
            "supported_adapter_ids": sorted(self._evaluators),
            "requires_same_adapter": False,
        }

    def validate_request(self, request: SuperOfflineComparisonRequest) -> None:
        self.validate_prepared_request(request, None)
        try:
            prepare_superoffline_input(request, repo_root=self._repo_root)
        except SuperOfflinePreflightError as exc:
            raise SuperOfflineAdapterError(
                f"recorder observation contract is not usable for this comparison: {exc}"
            ) from exc

    def validate_prepared_request(
        self,
        request: SuperOfflineComparisonRequest,
        _prepared_input: PreparedSuperOfflineInput | None,
    ) -> None:
        unsupported = sorted(
            {
                model.adapter.id
                for model in request.models
                if model.adapter.id not in self._evaluators
            }
        )
        if unsupported:
            raise SuperOfflineAdapterError(
                "the built-in executor has no read-only evaluator for "
                + ", ".join(unsupported)
            )
        if request.observation_path is None:
            raise SuperOfflineAdapterError(
                "observation_path is required for a real-robot offline comparison"
            )
        for model in request.models:
            if model.adapter.id != "openpi":
                continue
            try:
                discover_openpi_norm_stats_assets(
                    self._resolved_checkpoint(model.checkpoint)
                )
            except (OpenPiCheckpointContractError, SuperOfflineExecutionError) as exc:
                raise SuperOfflineAdapterError(
                    f"OpenPi checkpoint is not usable by SuperOffline: {exc}"
                ) from exc

    def request_warnings(
        self, request: SuperOfflineComparisonRequest
    ) -> list[dict[str, Any]]:
        """Report semantic task mismatches without blocking offline inference."""
        runtime_task = str((request.metadata or {}).get("prompt", "")).strip()
        if not runtime_task:
            return []
        warnings: list[dict[str, Any]] = []
        for index, model in enumerate(request.models):
            if model.adapter.id != "fastwam":
                continue
            training_task = self._checkpoint_training_task(model.checkpoint)
            if training_task is None or _normalized_task(training_task) == _normalized_task(
                runtime_task
            ):
                continue
            warnings.append(
                {
                    "code": "task_mismatch",
                    "severity": "warning",
                    "model_index": index,
                    "checkpoint_label": model.label,
                    "requested_task": runtime_task,
                    "training_task": training_task,
                    "message": (
                        f"{model.label} was trained for {training_task!r}, but this "
                        f"run requests {runtime_task!r}; inference will continue."
                    ),
                }
            )
        return warnings

    def run(
        self,
        request: SuperOfflineComparisonRequest,
        contexts: Sequence[SuperOfflineRuntimeContext],
        *,
        event_callback: Callable[[Mapping[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
        prepared_input: PreparedSuperOfflineInput | None = None,
    ) -> Mapping[str, Any]:
        if prepared_input is None:
            try:
                prepared_input = prepare_superoffline_input(
                    request,
                    repo_root=self._repo_root,
                )
            except SuperOfflinePreflightError as exc:
                raise SuperOfflineExecutionError(str(exc)) from exc
        self.validate_prepared_request(request, prepared_input)
        if len(contexts) != len(request.models) or tuple(
            context.model for context in contexts
        ) != request.models:
            raise SuperOfflineExecutionError(
                "runtime contexts do not match the requested comparison models"
            )
        job_id = str(request.job_id or "")
        if _JOB_ID_RE.fullmatch(job_id) is None:
            raise SuperOfflineExecutionError("comparison job id is invalid")

        warnings = self.request_warnings(request)
        self._emit(
            event_callback,
            stage="start",
            message=(
                f"Starting {len(request.models)} checkpoints for the validated "
                "observation selection"
            ),
            progress=2,
        )
        self._raise_if_cancelled(cancel_event)

        metadata = dict(request.metadata or {})
        inference_indices = self._selected_inference_indices(
            prepared_input.root,
            metadata,
        )
        cache_key = self._result_cache_key(
            request,
            prepared_input.bindings,
            inference_indices,
        )
        cached_result = self._cached_result(cache_key)
        if cached_result is not None:
            self._emit(
                event_callback,
                stage="cache_hit",
                message=(
                    "Reusing an exact completed comparison; model loading and "
                    "inference were skipped"
                ),
                progress=100,
            )
            return cached_result

        run_dir = self._output_root / job_id
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            run_dir.mkdir()
        except FileExistsError as exc:
            raise SuperOfflineExecutionError(
                f"comparison output already exists: {run_dir}"
            ) from exc
        try:
            self._emit(
                event_callback,
                stage="snapshot",
                message="Creating immutable observation and calibration snapshots",
                progress=5,
            )
            observation_root = self._snapshot_observation_run(
                self._observation_root(str(request.observation_path)),
                run_dir,
            )
            snapshot_request = replace(request, observation_path=str(observation_root))
            _, recorder_contract, bindings = self._prepared_observation_bindings(
                snapshot_request
            )
            action_calibrations = tuple(
                self._action_calibration_from_binding(binding) for binding in bindings
            )
            action_calibration_snapshot = snapshot_action_calibrations(
                tuple(
                    (self._model_action_calibration(model), calibration)
                    for model, calibration in zip(request.models, action_calibrations)
                ),
                run_dir / "action_calibration_snapshot",
                repo_root=self._repo_root,
            )
        except (ActionCalibrationError, SuperOfflineAdapterError) as exc:
            raise SuperOfflineExecutionError(
                f"could not create stable comparison input snapshots: {exc}"
            ) from exc
        bindings_path = run_dir / "observation_bindings.json"
        try:
            write_observation_bindings(bindings_path, bindings)
        except ObservationContractError as exc:
            raise SuperOfflineExecutionError(
                f"could not write trusted observation bindings: {exc}"
            ) from exc

        observation_label = ", ".join(str(index) for index in inference_indices)
        self._emit(
            event_callback,
            stage="preprocess",
            message=f"Selected observation indices: {observation_label}",
            progress=9,
            latest_observation=observation_label,
        )
        model_output_dirs: list[list[Path]] = []
        model_count = len(request.models)
        for index, (model, context) in enumerate(zip(request.models, contexts)):
            self._raise_if_cancelled(cancel_event)
            model_start = 10 + int(index * 84 / model_count)
            model_end = 10 + int((index + 1) * 84 / model_count)
            self._emit(
                event_callback,
                stage="runtime_doctor",
                message=f"Checking runtime profile for checkpoint {index + 1}: {model.label}",
                progress=model_start,
                latest_checkpoint=model.label,
                latest_observation=observation_label,
            )
            self._doctor(
                context,
                run_dir,
                index,
                event_callback=event_callback,
                cancel_event=cancel_event,
                progress=model_start,
            )
            model_root = run_dir / (
                f"model_{index + 1}_{model.variant_id or model.adapter.id}"
            )
            self._emit(
                event_callback,
                stage="infer_action",
                message=(
                    f"Running checkpoint {index + 1}/{model_count}: {model.label} "
                    f"on {len(inference_indices)} observation chunk(s)"
                ),
                progress=min(model_start + 4, model_end),
                latest_checkpoint=model.label,
                latest_observation=observation_label,
            )
            self._run_model(
                context,
                observation_root=observation_root,
                output_dir=model_root,
                inference_indices=inference_indices,
                metadata=metadata,
                run_dir=run_dir,
                index=index,
                binding_path=bindings_path,
                binding_index=index,
                action_calibration_snapshot=action_calibration_snapshot,
                event_callback=event_callback,
                cancel_event=cancel_event,
                progress=min(model_start + 4, model_end),
            )
            model_output_dirs.append(
                [
                    model_root / "observations" / f"{inference_index:06d}"
                    for inference_index in inference_indices
                ]
            )
            self._emit(
                event_callback,
                stage="checkpoint_complete",
                message=f"Checkpoint {index + 1}/{model_count} completed: {model.label}",
                progress=model_end,
                latest_checkpoint=model.label,
                latest_observation=observation_label,
            )

        self._raise_if_cancelled(cancel_event)
        self._emit(
            event_callback,
            stage="aggregate",
            message="Validating action artifacts and computing pairwise metrics",
            progress=96,
            latest_observation=observation_label,
        )
        samples: list[dict[str, Any]] = []
        for sample_number, inference_index in enumerate(inference_indices):
            results: list[dict[str, Any]] = []
            actions: list[np.ndarray] = []
            for index, (model, binding) in enumerate(zip(request.models, bindings)):
                model_dir = model_output_dirs[index][sample_number]
                summary = self._read_summary(model_dir, model=model, binding=binding)
                actions.append(self._read_canonical_action(model_dir, summary))
                results.append(
                    {
                        "model": model.as_dict(),
                        "output_dir": model_dir.relative_to(run_dir).as_posix(),
                        "summary": summary,
                        "task_validation": self._task_validation(
                            model,
                            runtime_task=str(metadata.get("prompt", "")).strip(),
                        ),
                    }
                )

            observation_timestamp_ns = self._shared_observation_timestamp(results)
            intervals = self._action_step_intervals(results)
            comparisons = self._pairwise_comparisons(
                request.models,
                actions,
                intervals,
            )
            samples.append(
                {
                    "sample_number": sample_number,
                    "inference_index": inference_index,
                    "observation_timestamp_ns": observation_timestamp_ns,
                    "models": results,
                    "comparison": comparisons[0]["metrics"],
                    "comparisons": comparisons,
                }
            )

        primary = samples[0]

        manifest = {
            "schema": "superoffline.real-robot-comparison.v1",
            "job_id": job_id,
            "source_observation_path": str(request.observation_path),
            "observation_snapshot": observation_root.relative_to(run_dir).as_posix(),
            "action_calibration_snapshot": action_calibration_snapshot.relative_to(
                run_dir
            ).as_posix(),
            "recorder_source_profile": recorder_contract.source_profile,
            "source_provenance_digest": bindings[0].source_provenance_digest,
            "recording_artifact_digest": bindings[0].recording_artifact_digest,
            "action_calibrations": [
                calibration.as_dict() for calibration in action_calibrations
            ],
            "observation_bindings": bindings_path.name,
            "observation_count": len(samples),
            "observation_timestamp_ns": primary["observation_timestamp_ns"],
            "models": primary["models"],
            "comparison": primary["comparison"],
            "comparisons": primary["comparisons"],
            "samples": samples,
            "warnings": warnings,
            "cache": {"version": _RESULT_CACHE_VERSION, "key": cache_key},
        }
        manifest_path = run_dir / "comparison.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {
            "executor": "builtin-real-robot-offline",
            "output_dir": str(run_dir),
            "comparison_manifest": str(manifest_path),
            "observation_bindings": str(bindings_path),
            "observation_snapshot": str(observation_root),
            "observation_count": len(samples),
            "models": primary["models"],
            "comparison": primary["comparison"],
            "comparisons": primary["comparisons"],
            "samples": samples,
            "warnings": warnings,
            "cache_hit": False,
        }

    @staticmethod
    def _selected_inference_indices(
        observation_root: Path,
        metadata: Mapping[str, Any],
    ) -> list[int]:
        manifest = json.loads((observation_root / "manifest.json").read_text(encoding="utf-8"))
        try:
            count = int(manifest["inference_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SuperOfflineExecutionError(
                "observation manifest has no valid inference_count"
            ) from exc
        if count < 1:
            raise SuperOfflineExecutionError("observation run has no inference samples")
        scope = str(metadata.get("run_scope", "sample"))
        if scope == "all":
            return list(range(count))
        if scope == "stride":
            stride = _metadata_integer(
                metadata,
                "sample_stride",
                default=10,
                minimum=1,
            )
            return list(range(0, count, stride))
        if scope != "sample":
            raise SuperOfflineExecutionError(
                "metadata.run_scope must be all, sample, or stride"
            )
        selected = _metadata_integer(metadata, "inference_index", default=count - 1)
        if selected < 0:
            selected += count
        if selected < 0 or selected >= count:
            raise SuperOfflineExecutionError(
                f"metadata.inference_index {selected} is outside the recorded run"
            )
        return [selected]

    def _result_cache_key(
        self,
        request: SuperOfflineComparisonRequest,
        bindings: Sequence[ObservationBinding],
        inference_indices: Sequence[int],
    ) -> str:
        payload = {
            "version": _RESULT_CACHE_VERSION,
            "observation": {
                "source_provenance_digest": bindings[0].source_provenance_digest,
                "recording_artifact_digest": bindings[0].recording_artifact_digest,
                "binding_digests": [binding.digest for binding in bindings],
            },
            "inference_indices": [int(index) for index in inference_indices],
            "models": [
                {
                    "request": model.as_dict(),
                    "checkpoint_identity": self._checkpoint_identity(model.checkpoint),
                }
                for model in request.models
            ],
            "metadata": dict(request.metadata or {}),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _checkpoint_identity(self, checkpoint: Any) -> dict[str, Any]:
        path = Path(self._resolved_checkpoint(checkpoint))
        if not path.exists():
            return {"path": str(path), "exists": False}
        if path.is_file():
            return {"path": str(path), "files": [self._file_identity(path, path.parent)]}

        candidates: set[Path] = set()
        for relative in ("config.yaml", "dataset_statistics.json", "params/manifest"):
            candidate = path / relative
            if candidate.is_file():
                candidates.add(candidate)
        for pattern in (
            "checkpoints/weights/*.pt",
            "assets/*/norm_stats.json",
            "params/**/*",
        ):
            for candidate in path.glob(pattern):
                if candidate.is_file():
                    candidates.add(candidate)
        return {
            "path": str(path),
            "files": [
                self._file_identity(candidate, path)
                for candidate in sorted(candidates)
            ],
        }

    @staticmethod
    def _file_identity(path: Path, root: Path) -> dict[str, Any]:
        stat = path.stat()
        record: dict[str, Any] = {
            "path": path.relative_to(root).as_posix(),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if stat.st_size <= 2 * 1024 * 1024:
            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return record

    def _cached_result(self, cache_key: str) -> dict[str, Any] | None:
        if not self._output_root.is_dir():
            return None
        try:
            candidates = sorted(
                (
                    path
                    for path in self._output_root.iterdir()
                    if path.is_dir() and not path.name.startswith("_")
                ),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
        except OSError:
            return None
        for run_dir in candidates:
            manifest_path = run_dir / "comparison.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            cache = manifest.get("cache")
            if not isinstance(cache, Mapping) or cache.get("key") != cache_key:
                continue
            if int(cache.get("version", -1)) != _RESULT_CACHE_VERSION:
                continue
            if not self._cached_manifest_artifacts_exist(run_dir, manifest):
                continue
            source_job_id = str(manifest.get("job_id") or run_dir.name)
            return {
                "executor": "builtin-real-robot-offline",
                "output_dir": str(run_dir),
                "comparison_manifest": str(manifest_path),
                "observation_bindings": str(
                    run_dir / str(manifest["observation_bindings"])
                ),
                "observation_snapshot": str(
                    run_dir / str(manifest["observation_snapshot"])
                ),
                "observation_count": int(manifest["observation_count"]),
                "models": manifest["models"],
                "comparison": manifest["comparison"],
                "comparisons": manifest["comparisons"],
                "samples": manifest["samples"],
                "warnings": manifest.get("warnings", []),
                "cache_hit": True,
                "cached_from_job_id": source_job_id,
            }
        return None

    @staticmethod
    def _cached_manifest_artifacts_exist(
        run_dir: Path,
        manifest: Mapping[str, Any],
    ) -> bool:
        for key in ("observation_bindings", "observation_snapshot"):
            relative = manifest.get(key)
            if not isinstance(relative, str):
                return False
            resolved = (run_dir / relative).resolve()
            if not resolved.is_relative_to(run_dir.resolve()) or not resolved.exists():
                return False
        samples = manifest.get("samples")
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
            return False
        for sample in samples:
            if not isinstance(sample, Mapping):
                return False
            models = sample.get("models")
            if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
                return False
            for model in models:
                if not isinstance(model, Mapping):
                    return False
                relative = model.get("output_dir")
                summary = model.get("summary")
                if not isinstance(relative, str) or not isinstance(summary, Mapping):
                    return False
                model_dir = (run_dir / relative).resolve()
                artifacts = summary.get("artifacts")
                if not model_dir.is_relative_to(run_dir.resolve()) or not isinstance(
                    artifacts, Mapping
                ):
                    return False
                npz_name = artifacts.get("npz")
                if not isinstance(npz_name, str) or not (model_dir / npz_name).is_file():
                    return False
        return True

    def _observation_root(self, raw_path: str) -> Path:
        root = Path(raw_path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not root.is_dir() or not manifest_path.is_file():
            raise SuperOfflineAdapterError(
                "observation_path must be a completed recorder run with manifest.json"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SuperOfflineAdapterError(
                f"could not read observation manifest: {manifest_path}"
            ) from exc
        source = str(manifest.get("source", "")).strip().lower()
        source_type = str(manifest.get("source_type", "")).strip().lower()
        if "sim" in source or "sim" in source_type:
            raise SuperOfflineAdapterError(
                "simulation observation inputs are not accepted by SuperOffline"
            )
        if manifest.get("schema") != _RECORDER_SCHEMA and source != "read_only_device_shm":
            raise SuperOfflineAdapterError(
                "observation manifest must use the timestamped recorder contract"
            )
        stopped_at_ns = manifest.get("stopped_at_ns")
        try:
            completed = int(stopped_at_ns)
        except (TypeError, ValueError) as exc:
            raise SuperOfflineAdapterError(
                "observation_path must be a completed recorder run with stopped_at_ns"
            ) from exc
        if completed <= 0:
            raise SuperOfflineAdapterError(
                "observation_path must be a completed recorder run with stopped_at_ns"
            )
        for key, label, count_key in (
            ("frames_index", "frames", "frame_count"),
            ("inference_index", "inference", "inference_count"),
        ):
            raw_name = manifest.get(key, f"{label}.jsonl")
            index_name = Path(str(raw_name))
            if index_name.name != str(raw_name) or str(raw_name) in {"", "."}:
                raise SuperOfflineAdapterError(
                    f"observation manifest has an invalid {key} path"
                )
            index_path = root / index_name
            try:
                row_count = self._validated_index_row_count(
                    root,
                    index_path,
                    label=label,
                )
            except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
                raise SuperOfflineAdapterError(
                    f"observation run has no valid {label} index: {index_path}"
                ) from exc
            declared_count = manifest.get(count_key)
            try:
                expected_count = int(declared_count)
            except (TypeError, ValueError) as exc:
                raise SuperOfflineAdapterError(
                    f"observation manifest is missing {count_key}"
                ) from exc
            if expected_count <= 0 or row_count != expected_count:
                raise SuperOfflineAdapterError(
                    f"observation {label} index does not match its finalized manifest"
                )
        return root

    def _prepared_observation_bindings(
        self,
        request: SuperOfflineComparisonRequest,
    ) -> tuple[Path, Any, tuple[ObservationBinding, ...]]:
        """Verify roles and all artifacts before resolving or starting runtimes."""
        try:
            prepared = prepare_superoffline_input(request, repo_root=self._repo_root)
        except SuperOfflinePreflightError as exc:
            raise SuperOfflineAdapterError(
                f"recorder observation contract is not usable for this comparison: {exc}"
            ) from exc
        return prepared.root, prepared.recorder_contract, prepared.bindings

    def _checkpoint_root(self, checkpoint: Any) -> Path:
        resolved = Path(self._resolved_checkpoint(checkpoint))
        if resolved.is_file():
            if (
                resolved.parent.name == "weights"
                and resolved.parent.parent.name == "checkpoints"
            ):
                return resolved.parent.parent.parent
            return resolved.parent
        return resolved

    def _checkpoint_training_task(self, checkpoint: Any) -> str | None:
        config_path = self._checkpoint_root(checkpoint) / "config.yaml"
        if not config_path.is_file():
            return None
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return None
        if not isinstance(payload, Mapping):
            return None
        for path in (
            ("datasets", "vla_data", "override_instruction"),
            ("datasets", "vla_data", "task"),
            ("framework", "fastwam", "task"),
            ("task",),
        ):
            value = _nested_text(payload, *path)
            if value is not None:
                return value
        return None

    def _task_validation(
        self,
        model: Any,
        *,
        runtime_task: str,
    ) -> dict[str, Any]:
        if model.adapter.id != "fastwam":
            return {"status": "not_applicable", "requested_task": runtime_task or None}
        training_task = self._checkpoint_training_task(model.checkpoint)
        if not runtime_task:
            status = "runtime_task_unavailable"
        elif training_task is None:
            status = "training_task_unavailable"
        elif _normalized_task(training_task) == _normalized_task(runtime_task):
            status = "match"
        else:
            status = "mismatch_warning"
        return {
            "status": status,
            "requested_task": runtime_task or None,
            "training_task": training_task,
        }

    @staticmethod
    def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise SuperOfflineExecutionError("comparison cancelled by operator")

    @staticmethod
    def _emit(
        event_callback: Callable[[Mapping[str, Any]], None] | None,
        *,
        stage: str,
        message: str,
        progress: int | None = None,
        severity: str = "info",
        latest_checkpoint: str | None = None,
        latest_observation: str | None = None,
    ) -> None:
        event = {
            "stage": stage,
            "message": message,
            "severity": severity,
        }
        if progress is not None:
            event["progress"] = max(0, min(int(progress), 100))
        if latest_checkpoint is not None:
            event["latest_checkpoint"] = latest_checkpoint
        if latest_observation is not None:
            event["latest_observation"] = latest_observation
        if event_callback is not None:
            event_callback(event)
        else:
            print(f"[{stage}] {message}", flush=True)

    @staticmethod
    def _model_action_calibration(model: Any) -> ActionCalibrationContract:
        contract = getattr(model, "action_calibration", None)
        if not isinstance(contract, ActionCalibrationContract):
            raise SuperOfflineAdapterError(
                "selected model has no registered action calibration contract"
            )
        return contract

    def _validated_action_calibrations(
        self,
        request: SuperOfflineComparisonRequest,
    ) -> tuple[ActionCalibrationBinding, ...]:
        """Compose each exact config before signing its action conversion contract."""
        try:
            from incubator.control_panel.runtime_config import compose_runtime_config

            calibrations = []
            for model in request.models:
                config = compose_runtime_config(model.config_name)
                policy = config.get("policy")
                if not isinstance(policy, Mapping):
                    raise ActionCalibrationError(
                        f"{model.config_name!r} has no composed policy section"
                    )
                params = policy.get("params")
                if not isinstance(params, Mapping):
                    raise ActionCalibrationError(
                        f"{model.config_name!r} has no composed policy.params section"
                    )
                calibrations.append(
                    validate_action_calibration_config(
                        self._model_action_calibration(model),
                        params,
                        repo_root=self._repo_root,
                    )
                )
        except ActionCalibrationError:
            raise
        except Exception as exc:
            raise ActionCalibrationError(
                "could not compose a registered model action calibration contract"
            ) from exc
        if len(calibrations) != len(request.models):
            raise ActionCalibrationError(
                "SuperOffline action calibration count does not match the selected models"
            )
        return tuple(calibrations)

    @staticmethod
    def _action_calibration_from_binding(
        binding: ObservationBinding,
    ) -> ActionCalibrationBinding:
        return ActionCalibrationBinding(
            calibration_id=binding.action_calibration_id,
            file_digests=dict(binding.action_calibration_file_digests),
            digest=binding.action_calibration_digest,
        )

    @staticmethod
    def _snapshot_observation_run(source_root: Path, run_dir: Path) -> Path:
        """Copy only the finalized recorder payload so both models read one image set."""
        snapshot_root = run_dir / "recording_snapshot"
        try:
            snapshot_root.mkdir()
            manifest_path = source_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            shutil.copyfile(manifest_path, snapshot_root / "manifest.json")
            copied: set[Path] = {Path("manifest.json")}
            for manifest_key, default_name, label in (
                ("frames_index", "frames.jsonl", "frames"),
                ("inference_index", "inference.jsonl", "inference"),
            ):
                raw_index = manifest.get(manifest_key, default_name)
                relative_index = Path(str(raw_index))
                if relative_index.name != str(raw_index) or str(raw_index) in {"", "."}:
                    raise SuperOfflineAdapterError(
                        f"observation manifest has an invalid {manifest_key} path"
                    )
                source_index = source_root / relative_index
                if not source_index.is_file():
                    raise SuperOfflineAdapterError(
                        f"observation run has no valid {label} index: {source_index}"
                    )
                shutil.copyfile(source_index, snapshot_root / relative_index)
                copied.add(relative_index)
                for line in source_index.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise SuperOfflineAdapterError(
                            f"{label} index contains a non-object row"
                        )
                    raw_artifact = row.get("file")
                    relative_artifact = Path(str(raw_artifact or ""))
                    if (
                        relative_artifact.is_absolute()
                        or ".." in relative_artifact.parts
                        or len(relative_artifact.parts) < 2
                        or relative_artifact.parts[0] != label
                        or relative_artifact.suffix != ".npz"
                    ):
                        raise SuperOfflineAdapterError(
                            f"{label} index has an invalid artifact path"
                        )
                    if relative_artifact in copied:
                        continue
                    source_artifact = (source_root / relative_artifact).resolve()
                    if (
                        not source_artifact.is_relative_to(source_root.resolve())
                        or not source_artifact.is_file()
                    ):
                        raise SuperOfflineAdapterError(
                            f"{label} index artifact is missing or leaves observation root"
                        )
                    destination = snapshot_root / relative_artifact
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_artifact, destination)
                    copied.add(relative_artifact)
        except (OSError, json.JSONDecodeError) as exc:
            raise SuperOfflineAdapterError(
                f"could not snapshot finalized recorder input: {source_root}"
            ) from exc
        return snapshot_root

    @staticmethod
    def _validated_index_row_count(
        root: Path,
        path: Path,
        *,
        label: str,
    ) -> int:
        expected_kind = "frame" if label == "frames" else "inference"
        count = 0
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise json.JSONDecodeError("index row must be an object", line, 0)
                if row.get("kind") != expected_kind:
                    raise ValueError(f"{label} index row has invalid kind")
                try:
                    timestamp_ns = int(row["timestamp_ns"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{label} index row has no valid timestamp") from exc
                if timestamp_ns <= 0:
                    raise ValueError(f"{label} index row has no valid timestamp")
                raw_file = row.get("file")
                if not isinstance(raw_file, str) or not raw_file.strip():
                    raise ValueError(f"{label} index row has no artifact path")
                relative_file = Path(raw_file)
                if (
                    relative_file.is_absolute()
                    or ".." in relative_file.parts
                    or len(relative_file.parts) < 2
                    or relative_file.parts[0] != label
                    or relative_file.suffix != ".npz"
                ):
                    raise ValueError(f"{label} index row has an invalid artifact path")
                artifact_path = (root / relative_file).resolve()
                if not artifact_path.is_relative_to(root.resolve()):
                    raise ValueError(f"{label} index artifact leaves observation root")
                if not artifact_path.is_file():
                    raise ValueError(f"{label} index artifact is missing")
                with np.load(artifact_path, allow_pickle=False) as artifact:
                    if not artifact.files:
                        raise ValueError(f"{label} index artifact has no arrays")
                count += 1
        return count

    def _doctor(
        self,
        context: SuperOfflineRuntimeContext,
        run_dir: Path,
        index: int,
        *,
        event_callback: Callable[[Mapping[str, Any]], None] | None,
        cancel_event: threading.Event | None,
        progress: int,
    ) -> None:
        # The diagnostics wrapper is launched with the service interpreter.
        # Its ``doctor``/``run`` commands resolve the verified profile again;
        # the evaluator itself is therefore executed by context.python, not
        # by the browser service interpreter.
        command = [
            sys.executable,
            str(self._runtime_profiles),
            "doctor",
            "--role",
            context.model.adapter.runtime_role,
            "--config-name",
            context.model.config_name,
        ]
        self._invoke(
            command,
            environment=context.environment,
            log_path=run_dir / f"model_{index + 1}_doctor.log",
            stage="runtime profile doctor",
            event_callback=event_callback,
            cancel_event=cancel_event,
            progress=progress,
        )

    def _run_model(
        self,
        context: SuperOfflineRuntimeContext,
        *,
        observation_root: Path,
        output_dir: Path,
        inference_indices: Sequence[int],
        metadata: Mapping[str, Any],
        run_dir: Path,
        index: int,
        binding_path: Path,
        binding_index: int,
        action_calibration_snapshot: Path,
        event_callback: Callable[[Mapping[str, Any]], None] | None,
        cancel_event: threading.Event | None,
        progress: int,
    ) -> None:
        model = context.model
        try:
            evaluator = self._evaluators[model.adapter.id]
        except KeyError as exc:
            raise SuperOfflineExecutionError(
                f"no built-in evaluator is registered for {model.adapter.id!r}"
            ) from exc
        evaluator_args = self._evaluator_args(
            model.adapter.id,
            model=model,
            observation_root=observation_root,
            output_dir=output_dir,
            inference_indices=inference_indices,
            metadata=metadata,
            binding_path=binding_path,
            binding_index=binding_index,
            action_calibration_snapshot=action_calibration_snapshot,
        )
        command = [
            sys.executable,
            str(self._runtime_profiles),
            "run",
            "--role",
            model.adapter.runtime_role,
            "--config-name",
            model.config_name,
            "--",
            str(evaluator),
            *evaluator_args,
        ]
        self._invoke(
            command,
            environment=context.environment,
            log_path=run_dir / f"model_{index + 1}_worker.log",
            stage=f"{model.adapter.label} offline evaluation for {model.label}",
            event_callback=event_callback,
            cancel_event=cancel_event,
            progress=progress,
        )

    def _evaluator_args(
        self,
        adapter_id: str,
        *,
        model: Any,
        observation_root: Path,
        output_dir: Path,
        inference_indices: Sequence[int],
        metadata: Mapping[str, Any],
        binding_path: Path,
        binding_index: int,
        action_calibration_snapshot: Path,
    ) -> list[str]:
        arguments = [
            "--config-name",
            model.config_name,
            "--checkpoint",
            self._resolved_checkpoint(model.checkpoint),
            "--observation-dir",
            str(observation_root),
            "--inference-indices",
            ",".join(str(index) for index in inference_indices),
            "--seed",
            str(_metadata_integer(metadata, "seed", default=42)),
            "--device",
            _safe_text(metadata.get("device", "cuda"), field="device"),
            "--output-dir",
            str(output_dir),
            "--observation-binding",
            str(binding_path),
            "--binding-index",
            str(binding_index),
            "--action-calibration-snapshot",
            str(action_calibration_snapshot),
        ]
        if adapter_id == "fastwam":
            output_mode = str(metadata.get("output_mode", "trajectory_only"))
            if output_mode not in {"trajectory_only", "trajectory_video"}:
                raise SuperOfflineExecutionError(
                    "metadata.output_mode must be trajectory_only or trajectory_video"
                )
            arguments[8:8] = [
                "--num-inference-steps",
                str(
                    _metadata_integer(
                        metadata,
                        "num_inference_steps",
                        default=10,
                        minimum=1,
                    )
                ),
                "--output-mode",
                output_mode,
            ]
            arguments.append("--skip-diagnostic-plots")
        prompt = metadata.get("prompt")
        if prompt not in (None, ""):
            arguments.extend(["--prompt", _safe_text(prompt, field="prompt")])
        return arguments

    def _resolved_checkpoint(self, checkpoint: Any) -> str:
        """Anchor relative checkpoint input before a runtime profile changes CWD."""
        raw_checkpoint = _safe_text(checkpoint, field="checkpoint")
        path = Path(raw_checkpoint).expanduser()
        if path.is_absolute():
            return str(path.resolve())
        resolved = (self._repo_root / path).resolve()
        if not resolved.is_relative_to(self._repo_root):
            raise SuperOfflineExecutionError(
                "relative checkpoint paths must remain within the repository"
            )
        return str(resolved)

    def _invoke(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str],
        log_path: Path,
        stage: str,
        event_callback: Callable[[Mapping[str, Any]], None] | None,
        cancel_event: threading.Event | None,
        progress: int,
    ) -> None:
        started_at = time.monotonic()
        self._emit(
            event_callback,
            stage="process_start",
            message=f"Starting {stage}",
            progress=progress,
        )
        if self._command_runner is subprocess.run:
            self._invoke_streaming(
                command,
                environment=environment,
                log_path=log_path,
                stage=stage,
                event_callback=event_callback,
                cancel_event=cancel_event,
                progress=progress,
            )
            self._emit(
                event_callback,
                stage="process_complete",
                message=f"Finished {stage} in {time.monotonic() - started_at:.2f} s",
                progress=progress,
            )
            return
        completed = self._command_runner(
            list(command),
            cwd=self._repo_root,
            env=dict(environment),
            check=False,
            capture_output=True,
            text=True,
        )
        stdout = str(getattr(completed, "stdout", "") or "")
        stderr = str(getattr(completed, "stderr", "") or "")
        log_path.write_text(
            f"$ {' '.join(command)}\n\n[stdout]\n{stdout}\n[stderr]\n{stderr}",
            encoding="utf-8",
        )
        for line in (stdout + "\n" + stderr).splitlines():
            if line.strip():
                self._emit(
                    event_callback,
                    stage=self._runtime_output_stage(stage, line),
                    message=self._public_log_line(line),
                    progress=progress,
                )
        self._raise_if_cancelled(cancel_event)
        if int(getattr(completed, "returncode", 1)) != 0:
            detail = (stderr or stdout).strip().replace("\n", " ")[:500]
            raise SuperOfflineExecutionError(
                f"{stage} failed with exit code {completed.returncode}: {detail}"
            )
        self._emit(
            event_callback,
            stage="process_complete",
            message=f"Finished {stage} in {time.monotonic() - started_at:.2f} s",
            progress=progress,
        )

    def _invoke_streaming(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str],
        log_path: Path,
        stage: str,
        event_callback: Callable[[Mapping[str, Any]], None] | None,
        cancel_event: threading.Event | None,
        progress: int,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            list(command),
            cwd=self._repo_root,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        lines: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        tail: list[str] = []
        cancelled = False
        cancel_requested_at: float | None = None
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"$ {' '.join(command)}\n\n[combined output]\n")
            log_file.flush()
            stream_finished = False
            while not stream_finished or process.poll() is None:
                if cancel_event is not None and cancel_event.is_set() and not cancelled:
                    cancelled = True
                    cancel_requested_at = time.monotonic()
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                if (
                    cancelled
                    and cancel_requested_at is not None
                    and process.poll() is None
                    and time.monotonic() - cancel_requested_at >= 5.0
                ):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                try:
                    item = lines.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    stream_finished = True
                    continue
                log_file.write(item)
                log_file.flush()
                text = item.rstrip("\r\n")
                if not text:
                    continue
                tail.append(text)
                tail = tail[-20:]
                self._emit(
                    event_callback,
                    stage=self._runtime_output_stage(stage, text),
                    message=self._public_log_line(text),
                    progress=progress,
                )
            if cancelled and process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            returncode = process.wait()
        reader.join(timeout=1)
        if cancelled:
            raise SuperOfflineExecutionError("comparison cancelled by operator")
        if returncode != 0:
            detail = " ".join(tail).strip()[-500:] or "no process output"
            raise SuperOfflineExecutionError(
                f"{stage} failed with exit code {returncode}: {detail}"
            )

    def _public_log_line(self, line: str) -> str:
        text = _ANSI_ESCAPE_RE.sub("", line).strip()
        text = text.replace(str(self._repo_root), "<repo>")
        home = str(Path.home())
        if home:
            text = text.replace(home, "~")
        return text[:800]

    @staticmethod
    def _runtime_output_stage(process_stage: str, line: str) -> str:
        if process_stage == "runtime profile doctor":
            return "runtime_doctor"
        text = line.casefold()
        if "unload" in text or "cleanup" in text or "empty_cache" in text:
            return "unload_checkpoint"
        if "encode" in text and ("video" in text or "frame" in text):
            return "encode_video"
        if "infer_joint" in text or "video prediction" in text:
            return "infer_joint"
        if "prompt" in text or "text encoder" in text or "t5" in text:
            return "prompt_conditioning"
        if "preprocess" in text or "model input" in text:
            return "preprocess"
        if "load" in text and ("checkpoint" in text or "weight" in text or "model" in text):
            return "load_checkpoint"
        if "infer" in text or "sampling" in text or "action" in text:
            return "infer_action"
        return "runtime_output"

    @staticmethod
    def _read_summary(
        output_dir: Path,
        *,
        model: Any,
        binding: ObservationBinding,
    ) -> dict[str, Any]:
        summary_path = output_dir / "summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SuperOfflineExecutionError(
                f"offline evaluator did not produce a valid summary: {summary_path}"
            ) from exc
        shape = summary.get("action_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or int(shape[0]) < 1
            or int(shape[1]) != 16
        ):
            raise SuperOfflineExecutionError(
                "offline evaluator output does not satisfy action [T,16]"
            )
        if summary.get("finite") is not True:
            raise SuperOfflineExecutionError("offline evaluator reported non-finite actions")
        if summary.get("action_contract") != DUAL_ARM_TCP_ACTION_CONTRACT:
            raise SuperOfflineExecutionError(
                "offline evaluator output does not declare the dual-arm TCP action contract"
            )
        if summary.get("action_layout") != DUAL_ARM_TCP_ACTION_LAYOUT:
            raise SuperOfflineExecutionError(
                "offline evaluator output does not declare the canonical TCP layout"
            )
        expected_action_frame = str(
            getattr(model, "action_frame_id", DUAL_ARM_TCP_ACTION_FRAME)
        )
        expected_action_units = str(
            getattr(model, "action_units", DUAL_ARM_TCP_ACTION_UNITS)
        )
        if summary.get("action_frame_id") != expected_action_frame:
            raise SuperOfflineExecutionError(
                "offline evaluator output does not declare the reviewed TCP action frame"
            )
        if summary.get("action_units") != expected_action_units:
            raise SuperOfflineExecutionError(
                "offline evaluator output does not declare the reviewed TCP action units"
            )
        if summary.get("observation_binding_digest") != binding.digest:
            raise SuperOfflineExecutionError(
                "offline evaluator output was not produced from the trusted observation binding"
            )
        if summary.get("source_profile") != binding.source_profile:
            raise SuperOfflineExecutionError(
                "offline evaluator output source profile does not match the observation binding"
            )
        if summary.get("source_provenance_digest") != binding.source_provenance_digest:
            raise SuperOfflineExecutionError(
                "offline evaluator output source provenance does not match the observation binding"
            )
        if summary.get("recording_artifact_digest") != binding.recording_artifact_digest:
            raise SuperOfflineExecutionError(
                "offline evaluator output recorder artifacts do not match the observation binding"
            )
        if summary.get("action_calibration_id") != binding.action_calibration_id:
            raise SuperOfflineExecutionError(
                "offline evaluator output action calibration does not match the observation binding"
            )
        if summary.get("action_calibration_digest") != binding.action_calibration_digest:
            raise SuperOfflineExecutionError(
                "offline evaluator output action calibration digest does not match the observation binding"
            )
        artifacts = summary.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise SuperOfflineExecutionError(
                "offline evaluator summary is missing its action artifact contract"
            )
        if artifacts.get("action_contract") != DUAL_ARM_TCP_ACTION_CONTRACT:
            raise SuperOfflineExecutionError(
                "offline evaluator artifact does not declare the dual-arm TCP action contract"
            )
        if artifacts.get("action_layout") != DUAL_ARM_TCP_ACTION_LAYOUT:
            raise SuperOfflineExecutionError(
                "offline evaluator artifact does not declare the canonical TCP layout"
            )
        if artifacts.get("action_frame_id") != expected_action_frame:
            raise SuperOfflineExecutionError(
                "offline evaluator artifact does not declare the reviewed TCP action frame"
            )
        if artifacts.get("action_units") != expected_action_units:
            raise SuperOfflineExecutionError(
                "offline evaluator artifact does not declare the reviewed TCP action units"
            )
        if artifacts.get("observation_binding_digest") != binding.digest:
            raise SuperOfflineExecutionError(
                "offline evaluator artifact does not declare the trusted observation binding"
            )
        if artifacts.get("source_provenance_digest") != binding.source_provenance_digest:
            raise SuperOfflineExecutionError(
                "offline evaluator artifact does not declare the reviewed source provenance"
            )
        if artifacts.get("recording_artifact_digest") != binding.recording_artifact_digest:
            raise SuperOfflineExecutionError(
                "offline evaluator artifact does not declare the recorder artifact digest"
            )
        if artifacts.get("action_calibration_id") != binding.action_calibration_id:
            raise SuperOfflineExecutionError(
                "offline evaluator artifact does not declare the reviewed action calibration"
            )
        if artifacts.get("action_calibration_digest") != binding.action_calibration_digest:
            raise SuperOfflineExecutionError(
                "offline evaluator artifact does not declare the reviewed action calibration digest"
            )
        RealRobotOfflineComparisonExecutor._summary_action_step_interval(summary)
        artifact_interval = artifacts.get("action_step_interval_ns")
        if artifact_interval != summary.get("action_step_interval_ns"):
            raise SuperOfflineExecutionError(
                "offline evaluator action timebase differs between summary and artifact"
            )
        return summary

    @staticmethod
    def _read_canonical_action(
        output_dir: Path,
        summary: Mapping[str, Any],
    ) -> np.ndarray:
        """Load the normalized TCP action artifact declared by an evaluator."""
        artifacts = summary.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise SuperOfflineExecutionError(
                "offline evaluator summary is missing its action artifact contract"
            )
        npz_name = artifacts.get("npz")
        action_key = artifacts.get("canonical_action_key")
        if not isinstance(npz_name, str) or not npz_name:
            raise SuperOfflineExecutionError("offline evaluator summary is missing artifacts.npz")
        if not isinstance(action_key, str) or not action_key:
            raise SuperOfflineExecutionError(
                "offline evaluator summary is missing artifacts.canonical_action_key"
            )
        output_root = output_dir.resolve()
        action_path = (output_root / npz_name).resolve()
        if not action_path.is_relative_to(output_root):
            raise SuperOfflineExecutionError("offline evaluator action artifact leaves its output directory")
        try:
            with np.load(action_path, allow_pickle=False) as values:
                action = np.asarray(values[action_key], dtype=np.float64)
        except (KeyError, OSError, ValueError) as exc:
            raise SuperOfflineExecutionError(
                f"could not load canonical action artifact: {action_path}"
            ) from exc
        if action.ndim != 2 or action.shape[0] < 1 or action.shape[1] != 16:
            raise SuperOfflineExecutionError(
                f"canonical action artifact must be [T,16], got {action.shape}"
            )
        if not np.isfinite(action).all():
            raise SuperOfflineExecutionError(
                "canonical action artifact contains non-finite values"
            )
        return action

    @staticmethod
    def _shared_observation_timestamp(results: Sequence[Mapping[str, Any]]) -> int:
        timestamps: list[int] = []
        for result in results:
            summary = result.get("summary")
            if not isinstance(summary, Mapping):
                raise SuperOfflineExecutionError("offline evaluator result is missing summary")
            try:
                timestamp = int(summary["observation_timestamp_ns"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SuperOfflineExecutionError(
                    "offline evaluator summary is missing observation_timestamp_ns"
                ) from exc
            if timestamp <= 0:
                raise SuperOfflineExecutionError(
                    "offline evaluator reported an invalid observation timestamp"
                )
            timestamps.append(timestamp)
        if len(set(timestamps)) != 1:
            raise SuperOfflineExecutionError(
                "the comparison models evaluated different recorded observations"
            )
        return timestamps[0]

    @staticmethod
    def _summary_action_step_interval(summary: Mapping[str, Any]) -> int:
        value = summary.get("action_step_interval_ns")
        if isinstance(value, bool):
            raise SuperOfflineExecutionError(
                "offline evaluator summary is missing a positive action_step_interval_ns"
            )
        try:
            interval_ns = int(value)
        except (TypeError, ValueError) as exc:
            raise SuperOfflineExecutionError(
                "offline evaluator summary is missing a positive action_step_interval_ns"
            ) from exc
        if interval_ns <= 0:
            raise SuperOfflineExecutionError(
                "offline evaluator summary is missing a positive action_step_interval_ns"
            )
        return interval_ns

    @classmethod
    def _action_step_intervals(
        cls,
        results: Sequence[Mapping[str, Any]],
    ) -> tuple[int, ...]:
        intervals: list[int] = []
        for result in results:
            summary = result.get("summary")
            if not isinstance(summary, Mapping):
                raise SuperOfflineExecutionError("offline evaluator result is missing summary")
            intervals.append(cls._summary_action_step_interval(summary))
        if len(intervals) < 2:
            raise SuperOfflineExecutionError(
                "comparison results must contain at least two action time bases"
            )
        return tuple(intervals)

    @classmethod
    def _pairwise_comparisons(
        cls,
        models: Sequence[Any],
        actions: Sequence[np.ndarray],
        intervals: Sequence[int],
    ) -> list[dict[str, Any]]:
        comparisons: list[dict[str, Any]] = []
        for first_index in range(len(models)):
            for second_index in range(first_index + 1, len(models)):
                comparisons.append(
                    {
                        "model_indices": [first_index, second_index],
                        "model_labels": [
                            models[first_index].label,
                            models[second_index].label,
                        ],
                        "metrics": cls._comparison_metrics(
                            actions[first_index],
                            actions[second_index],
                            first_action_step_interval_ns=intervals[first_index],
                            second_action_step_interval_ns=intervals[second_index],
                        ),
                    }
                )
        return comparisons

    @staticmethod
    def _comparison_metrics(
        first: np.ndarray,
        second: np.ndarray,
        *,
        first_action_step_interval_ns: int,
        second_action_step_interval_ns: int,
    ) -> dict[str, Any]:
        """Measure compatible TCP actions on a shared, physical time base."""
        first_horizon = int(first.shape[0])
        second_horizon = int(second.shape[0])
        first_duration_ns = (first_horizon - 1) * first_action_step_interval_ns
        second_duration_ns = (second_horizon - 1) * second_action_step_interval_ns
        compared_duration_ns = min(first_duration_ns, second_duration_ns)
        comparison_step_interval_ns = max(
            first_action_step_interval_ns,
            second_action_step_interval_ns,
        )
        compared_steps = int(compared_duration_ns // comparison_step_interval_ns) + 1
        sample_times_ns = (
            np.arange(compared_steps, dtype=np.int64) * comparison_step_interval_ns
        )
        first = RealRobotOfflineComparisonExecutor._resample_tcp_action(
            first,
            action_step_interval_ns=first_action_step_interval_ns,
            sample_times_ns=sample_times_ns,
        )
        second = RealRobotOfflineComparisonExecutor._resample_tcp_action(
            second,
            action_step_interval_ns=second_action_step_interval_ns,
            sample_times_ns=sample_times_ns,
        )

        def stats(values: np.ndarray) -> dict[str, float]:
            return {
                "mean": float(np.mean(values)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)),
            }

        arms: dict[str, dict[str, dict[str, float]]] = {}
        for arm_name, offset in (("left", 0), ("right", 8)):
            position_error = np.linalg.norm(
                first[:, offset : offset + 3] - second[:, offset : offset + 3],
                axis=1,
            )
            first_quaternion = first[:, offset + 3 : offset + 7]
            second_quaternion = second[:, offset + 3 : offset + 7]
            first_norm = np.linalg.norm(first_quaternion, axis=1)
            second_norm = np.linalg.norm(second_quaternion, axis=1)
            if np.any(first_norm < 1e-8) or np.any(second_norm < 1e-8):
                raise SuperOfflineExecutionError(
                    f"{arm_name} action contains a zero-length TCP quaternion"
                )
            cosine = np.sum(
                first_quaternion / first_norm[:, None]
                * second_quaternion / second_norm[:, None],
                axis=1,
            )
            orientation_error = np.degrees(
                2.0 * np.arccos(np.clip(np.abs(cosine), 0.0, 1.0))
            )
            gripper_error = np.abs(first[:, offset + 7] - second[:, offset + 7])
            arms[arm_name] = {
                "position_error_m": stats(position_error),
                "orientation_error_deg": stats(orientation_error),
                "gripper_error_m": stats(gripper_error),
            }
        return {
            "schema": "superoffline.dual-arm-tcp-action-delta.v1",
            "first_horizon": first_horizon,
            "second_horizon": second_horizon,
            "steps_compared": int(compared_steps),
            "first_action_step_interval_ns": first_action_step_interval_ns,
            "second_action_step_interval_ns": second_action_step_interval_ns,
            "comparison_step_interval_ns": comparison_step_interval_ns,
            "compared_duration_ns": compared_duration_ns,
            "resampled": first_action_step_interval_ns != second_action_step_interval_ns,
            "arms": arms,
        }

    @staticmethod
    def _resample_tcp_action(
        action: np.ndarray,
        *,
        action_step_interval_ns: int,
        sample_times_ns: np.ndarray,
    ) -> np.ndarray:
        """Linearly sample translation/gripper and SLERP TCP quaternions."""
        source = np.asarray(action, dtype=np.float64)
        source_times_ns = np.arange(source.shape[0], dtype=np.int64) * int(
            action_step_interval_ns
        )
        if sample_times_ns.size == 0 or sample_times_ns[-1] > source_times_ns[-1]:
            raise SuperOfflineExecutionError("comparison sample times exceed action horizon")
        upper = np.searchsorted(source_times_ns, sample_times_ns, side="left")
        upper = np.clip(upper, 0, source.shape[0] - 1)
        lower = np.maximum(upper - 1, 0)
        span_ns = source_times_ns[upper] - source_times_ns[lower]
        fractions = np.divide(
            sample_times_ns - source_times_ns[lower],
            span_ns,
            out=np.zeros(sample_times_ns.shape, dtype=np.float64),
            where=span_ns > 0,
        )
        result = np.empty((sample_times_ns.size, 16), dtype=np.float64)
        for offset in (0, 8):
            result[:, offset : offset + 3] = (
                source[lower, offset : offset + 3]
                + fractions[:, None]
                * (
                    source[upper, offset : offset + 3]
                    - source[lower, offset : offset + 3]
                )
            )
            result[:, offset + 3 : offset + 7] = (
                RealRobotOfflineComparisonExecutor._slerp_quaternions(
                    source[lower, offset + 3 : offset + 7],
                    source[upper, offset + 3 : offset + 7],
                    fractions,
                )
            )
            result[:, offset + 7] = (
                source[lower, offset + 7]
                + fractions
                * (source[upper, offset + 7] - source[lower, offset + 7])
            )
        return result

    @staticmethod
    def _slerp_quaternions(
        first: np.ndarray,
        second: np.ndarray,
        fractions: np.ndarray,
    ) -> np.ndarray:
        first_norm = np.linalg.norm(first, axis=1)
        second_norm = np.linalg.norm(second, axis=1)
        if np.any(first_norm < 1e-8) or np.any(second_norm < 1e-8):
            raise SuperOfflineExecutionError(
                "action contains a zero-length TCP quaternion during resampling"
            )
        first_unit = first / first_norm[:, None]
        second_unit = second / second_norm[:, None]
        dot = np.sum(first_unit * second_unit, axis=1)
        second_unit = np.where((dot < 0.0)[:, None], -second_unit, second_unit)
        dot = np.abs(dot)
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta = np.sin(theta)
        first_weight = np.cos(fractions * theta)
        second_weight = np.sin(fractions * theta)
        regular = np.abs(sin_theta) > 1e-8
        first_weight[regular] = (
            np.sin((1.0 - fractions[regular]) * theta[regular])
            / sin_theta[regular]
        )
        second_weight[regular] = (
            np.sin(fractions[regular] * theta[regular]) / sin_theta[regular]
        )
        result = first_weight[:, None] * first_unit + second_weight[:, None] * second_unit
        result_norm = np.linalg.norm(result, axis=1)
        if np.any(result_norm < 1e-8):
            raise SuperOfflineExecutionError("could not normalize resampled TCP quaternion")
        return result / result_norm[:, None]


class FastWAMOfflineComparisonExecutor(RealRobotOfflineComparisonExecutor):
    """Compatibility name for integrations created before OpenPi support."""


__all__ = [
    "FastWAMOfflineComparisonExecutor",
    "RealRobotOfflineComparisonExecutor",
]
