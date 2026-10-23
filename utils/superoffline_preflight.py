"""Shared trusted-input preflight for real-robot SuperOffline jobs.

The browser service and every executor use this module before selecting a
runtime.  It binds one finalized recorder run to both composed policy configs,
so an alternate executor cannot silently reinterpret camera order, source rig,
or action-calibration files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from utils.superoffline_action_calibration import (
    ActionCalibrationBinding,
    ActionCalibrationContract,
    ActionCalibrationError,
    validate_action_calibration_config,
)
from utils.hydra_config import compose_hydra_application_config
from utils.superoffline_observation import (
    ObservationBinding,
    ObservationContractError,
    RecorderObservationContract,
    create_observation_binding,
    parse_recorder_observation_contract,
    validate_recording_artifacts,
    verify_source_profile_calibrations,
)


RECORDER_SCHEMA = "superinference.timestamped_observation"
_ROLE_ORDER = (
    "left_wrist_rgb",
    "right_wrist_rgb",
    "rizon_dual_arm_state",
)


class SuperOfflinePreflightError(ValueError):
    """A model pair or recorder input cannot form a trusted comparison job."""


@dataclass(frozen=True)
class PreparedSuperOfflineInput:
    """Validated source material shared by the two evaluator processes."""

    root: Path
    manifest: Mapping[str, Any]
    recorder_contract: RecorderObservationContract
    bindings: tuple[ObservationBinding, ObservationBinding]
    action_calibrations: tuple[ActionCalibrationBinding, ActionCalibrationBinding]


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SuperOfflinePreflightError(f"{field} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, *, field: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SuperOfflinePreflightError(f"{field} must be an array")
    result = [_text(item, field=field) for item in value]
    if not result:
        raise SuperOfflinePreflightError(f"{field} must not be empty")
    return result


def _model_target_devices(model: Any) -> list[str]:
    target_device_roles = getattr(model, "target_device_roles", None)
    if not isinstance(target_device_roles, Mapping):
        raise SuperOfflinePreflightError(
            "selected model has no registered target device-role contract"
        )
    targets_by_role: dict[str, str] = {}
    for raw_target, raw_role in target_device_roles.items():
        target = _text(raw_target, field="registered target device")
        role = _text(raw_role, field="registered target device role")
        if role in targets_by_role:
            raise SuperOfflinePreflightError(
                "registered target device roles must be one-to-one"
            )
        targets_by_role[role] = target
    if set(targets_by_role) != set(_ROLE_ORDER):
        raise SuperOfflinePreflightError(
            "registered target device roles must contain left/right camera and robot"
        )
    return [targets_by_role[role] for role in _ROLE_ORDER]


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SuperOfflinePreflightError(f"{field} must be an object")
    return value


def _validate_composed_observation_devices(
    model: Any,
    policy: Mapping[str, Any],
) -> None:
    """Ensure the evaluator's configured device ordering matches its binding."""
    expected_devices = _model_target_devices(model)
    config_name = _text(getattr(model, "config_name", None), field="config_name")
    observation = _mapping(policy.get("obs"), field=f"{config_name}.policy.obs")
    actual_observation_devices = _string_list(
        observation.get("devices"), field=f"{config_name}.policy.obs.devices"
    )
    if actual_observation_devices != expected_devices:
        raise SuperOfflinePreflightError(
            f"{config_name}.policy.obs.devices must preserve the registered "
            f"left/right/robot order {expected_devices!r}, got "
            f"{actual_observation_devices!r}"
        )

    adapter = getattr(getattr(model, "adapter", None), "id", None)
    if adapter not in {"fastwam", "lingbotva"}:
        return
    params = _mapping(policy.get("params"), field=f"{config_name}.policy.params")
    actual_camera_devices = _string_list(
        params.get("camera_devices"),
        field=f"{config_name}.policy.params.camera_devices",
    )
    actual_robot_device = _text(
        params.get("robot_device"),
        field=f"{config_name}.policy.params.robot_device",
    )
    if actual_camera_devices != expected_devices[:2] or actual_robot_device != expected_devices[2]:
        raise SuperOfflinePreflightError(
            f"{config_name}.policy.params camera/robot devices must preserve the "
            "registered left/right/robot order"
        )


def _model_action_calibration(model: Any) -> ActionCalibrationContract:
    contract = getattr(model, "action_calibration", None)
    if not isinstance(contract, ActionCalibrationContract):
        raise SuperOfflinePreflightError(
            "selected model has no registered action calibration contract"
        )
    return contract


def validated_action_calibrations(
    request: Any,
    *,
    repo_root: Path | str | None = None,
) -> tuple[ActionCalibrationBinding, ActionCalibrationBinding]:
    """Compose exact configs and validate both device order and action semantics."""
    models = getattr(request, "models", ())
    if not isinstance(models, Sequence) or len(models) != 2:
        raise SuperOfflinePreflightError(
            "SuperOffline comparisons require exactly two registered models"
        )
    try:
        calibrations: list[ActionCalibrationBinding] = []
        for model in models:
            config_name = _text(getattr(model, "config_name", None), field="config_name")
            config = compose_hydra_application_config(
                config_name,
                config_dir=Path(__file__).resolve().parents[1] / "configs",
                job_name="superoffline_preflight",
            )
            policy = _mapping(config.get("policy"), field=f"{config_name}.policy")
            _validate_composed_observation_devices(model, policy)
            params = _mapping(
                policy.get("params"), field=f"{config_name}.policy.params"
            )
            calibrations.append(
                validate_action_calibration_config(
                    _model_action_calibration(model),
                    params,
                    repo_root=repo_root,
                )
            )
    except (ActionCalibrationError, SuperOfflinePreflightError):
        raise
    except Exception as exc:
        raise SuperOfflinePreflightError(
            "could not compose a registered model comparison contract"
        ) from exc
    return calibrations[0], calibrations[1]


def _validated_index_row_count(root: Path, path: Path, *, label: str) -> int:
    expected_kind = "frame" if label == "frames" else "inference"
    count = 0
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError("index row must be an object")
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


def _completed_observation_root(raw_path: Any) -> tuple[Path, Mapping[str, Any]]:
    root = Path(_text(raw_path, field="observation_path")).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise SuperOfflinePreflightError(
            "observation_path must be a completed recorder run with manifest.json"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SuperOfflinePreflightError(
            f"could not read observation manifest: {manifest_path}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise SuperOfflinePreflightError("observation manifest must be an object")
    source = str(manifest.get("source", "")).strip().lower()
    source_type = str(manifest.get("source_type", "")).strip().lower()
    if "sim" in source or "sim" in source_type:
        raise SuperOfflinePreflightError(
            "simulation observation inputs are not accepted by SuperOffline"
        )
    if manifest.get("schema") != RECORDER_SCHEMA and source != "read_only_device_shm":
        raise SuperOfflinePreflightError(
            "observation manifest must use the timestamped recorder contract"
        )
    try:
        stopped_at_ns = int(manifest.get("stopped_at_ns"))
    except (TypeError, ValueError) as exc:
        raise SuperOfflinePreflightError(
            "observation_path must be a completed recorder run with stopped_at_ns"
        ) from exc
    if stopped_at_ns <= 0:
        raise SuperOfflinePreflightError(
            "observation_path must be a completed recorder run with stopped_at_ns"
        )
    for key, label, count_key in (
        ("frames_index", "frames", "frame_count"),
        ("inference_index", "inference", "inference_count"),
    ):
        raw_name = manifest.get(key, f"{label}.jsonl")
        index_name = Path(str(raw_name))
        if index_name.name != str(raw_name) or str(raw_name) in {"", "."}:
            raise SuperOfflinePreflightError(
                f"observation manifest has an invalid {key} path"
            )
        index_path = root / index_name
        try:
            row_count = _validated_index_row_count(root, index_path, label=label)
        except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            raise SuperOfflinePreflightError(
                f"observation run has no valid {label} index: {index_path}"
            ) from exc
        try:
            declared_count = int(manifest.get(count_key))
        except (TypeError, ValueError) as exc:
            raise SuperOfflinePreflightError(
                f"observation manifest is missing {count_key}"
            ) from exc
        if declared_count <= 0 or row_count != declared_count:
            raise SuperOfflinePreflightError(
                f"observation {label} index does not match its finalized manifest"
            )
    return root, manifest


def prepare_superoffline_input(
    request: Any,
    *,
    repo_root: Path | str | None = None,
) -> PreparedSuperOfflineInput:
    """Verify one finalized recording against both registered model contracts."""
    if getattr(request, "observation_path", None) is None:
        raise SuperOfflinePreflightError(
            "observation_path is required for a real-robot offline comparison"
        )
    try:
        root, manifest = _completed_observation_root(request.observation_path)
        recorder_contract = parse_recorder_observation_contract(manifest)
        if recorder_contract.legacy_identity:
            raise ObservationContractError(
                "legacy recorder manifests lack reviewed source provenance"
            )
        validate_recording_artifacts(root, recorder_contract)
        assert recorder_contract.source_provenance is not None
        verify_source_profile_calibrations(
            recorder_contract.source_provenance,
            repo_root=repo_root,
        )
        calibrations = validated_action_calibrations(request, repo_root=repo_root)
        bindings = tuple(
            create_observation_binding(
                manifest=manifest,
                config_name=model.config_name,
                target_device_roles=dict(model.target_device_roles or {}),
                allowed_source_profiles=model.source_profiles,
                action_calibration_id=calibration.calibration_id,
                action_calibration_file_digests=calibration.file_digests,
                action_calibration_digest=calibration.digest,
                recording_root=root,
            )
            for model, calibration in zip(request.models, calibrations)
        )
    except (
        ActionCalibrationError,
        ObservationContractError,
        OSError,
        json.JSONDecodeError,
        SuperOfflinePreflightError,
    ) as exc:
        if isinstance(exc, SuperOfflinePreflightError):
            raise
        raise SuperOfflinePreflightError(
            f"recorder observation contract is not usable for this comparison: {exc}"
        ) from exc
    if len(bindings) != 2:
        raise SuperOfflinePreflightError(
            "SuperOffline comparisons require exactly two trusted observation bindings"
        )
    return PreparedSuperOfflineInput(
        root=root,
        manifest=manifest,
        recorder_contract=recorder_contract,
        bindings=(bindings[0], bindings[1]),
        action_calibrations=calibrations,
    )


__all__ = [
    "PreparedSuperOfflineInput",
    "RECORDER_SCHEMA",
    "SuperOfflinePreflightError",
    "prepare_superoffline_input",
    "validated_action_calibrations",
]
