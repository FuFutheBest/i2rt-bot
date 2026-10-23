"""Reviewed action-calibration contracts for SuperOffline model outputs.

The recorder provenance proves how an observation was captured.  This module
separately proves which calibration semantics turn one model's output into the
common Rizon TCP action frame.  Both the job executor and evaluator process
derive the same digest so a configuration change cannot silently alter the
meaning of a comparison result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

from utils.superoffline_observation import reviewed_calibration_files


ACTION_CALIBRATION_SCHEMA = "superoffline.action-calibration.v1"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ActionCalibrationError(ValueError):
    """The configured action conversion is not a reviewed SuperOffline contract."""


def _identifier(value: Any, *, field_name: str) -> str:
    result = str(value or "").strip()
    if _IDENTIFIER_RE.fullmatch(result) is None:
        raise ActionCalibrationError(
            f"{field_name} must contain only letters, digits, dot, underscore, or dash"
        )
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ActionCalibrationError(f"could not hash action calibration {path}") from exc
    return digest.hexdigest()


@dataclass(frozen=True)
class ActionCalibrationContract:
    """Code-owned requirements for converting a model output to TCP actions."""

    calibration_id: str
    file_parameters: Mapping[str, str] = field(default_factory=dict)
    absent_parameters: tuple[str, ...] = ()
    required_policy_values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        calibration_id = _identifier(self.calibration_id, field_name="action calibration id")
        parameters = {
            _identifier(parameter, field_name="action calibration parameter"): _identifier(
                label, field_name="action calibration label"
            )
            for parameter, label in self.file_parameters.items()
        }
        absent = tuple(
            _identifier(parameter, field_name="absent action calibration parameter")
            for parameter in self.absent_parameters
        )
        values = {
            _identifier(parameter, field_name="action calibration policy parameter"): value
            for parameter, value in self.required_policy_values.items()
        }
        if len(parameters) != len(set(parameters)):
            raise ActionCalibrationError("action calibration file parameters must be unique")
        if len(absent) != len(set(absent)):
            raise ActionCalibrationError("absent action calibration parameters must be unique")
        if set(parameters) & set(absent):
            raise ActionCalibrationError(
                "an action calibration parameter cannot be both required and absent"
            )
        if set(parameters) & set(values):
            raise ActionCalibrationError(
                "an action calibration parameter cannot be both a file and a scalar value"
            )
        if set(absent) & set(values):
            raise ActionCalibrationError(
                "an action calibration parameter cannot be both absent and a scalar value"
            )
        object.__setattr__(self, "calibration_id", calibration_id)
        object.__setattr__(self, "file_parameters", parameters)
        object.__setattr__(self, "absent_parameters", absent)
        object.__setattr__(self, "required_policy_values", values)

    def public_metadata(self) -> dict[str, Any]:
        return {
            "action_calibration_id": self.calibration_id,
            "file_parameters": dict(self.file_parameters),
            "absent_parameters": list(self.absent_parameters),
            "required_policy_values": dict(self.required_policy_values),
        }


@dataclass(frozen=True)
class ActionCalibrationBinding:
    """Digest of the reviewed action calibration selected for one model run."""

    calibration_id: str
    file_digests: Mapping[str, str]
    digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_calibration_id": self.calibration_id,
            "action_calibration_file_digests": dict(self.file_digests),
            "action_calibration_digest": self.digest,
        }


def _binding_from_file_digests(
    contract: ActionCalibrationContract,
    file_digests: Mapping[str, Any],
) -> ActionCalibrationBinding:
    expected_keys = {
        f"{parameter}:{label}"
        for parameter, label in contract.file_parameters.items()
    }
    normalized_digests = {
        str(key): str(value).strip().lower() for key, value in file_digests.items()
    }
    if set(normalized_digests) != expected_keys:
        raise ActionCalibrationError(
            "action calibration file digests do not match the registered parameters"
        )
    for key, digest in normalized_digests.items():
        if _SHA256_RE.fullmatch(digest) is None:
            raise ActionCalibrationError(
                f"action calibration digest for {key!r} must be SHA-256"
            )
    ordered_digests = {
        key: normalized_digests[key] for key in sorted(normalized_digests)
    }
    payload = {
        "schema": ACTION_CALIBRATION_SCHEMA,
        "action_calibration_id": contract.calibration_id,
        "file_digests": ordered_digests,
        "absent_parameters": sorted(contract.absent_parameters),
        "required_policy_values": dict(contract.required_policy_values),
    }
    return ActionCalibrationBinding(
        calibration_id=contract.calibration_id,
        file_digests=ordered_digests,
        digest=_digest(payload),
    )


def reviewed_action_calibration(
    contract: ActionCalibrationContract,
    *,
    repo_root: Path | str | None = None,
) -> ActionCalibrationBinding:
    """Hash the repository-owned calibration files named by one contract."""
    reviewed_files = reviewed_calibration_files(repo_root)
    file_digests: dict[str, str] = {}
    for parameter, label in sorted(contract.file_parameters.items()):
        try:
            path = reviewed_files[label]
        except KeyError as exc:
            raise ActionCalibrationError(
                f"action calibration {contract.calibration_id!r} references unknown "
                f"reviewed label {label!r}"
            ) from exc
        if not path.is_file():
            raise ActionCalibrationError(
                f"reviewed action calibration file for {label!r} is missing: {path}"
            )
        file_digests[f"{parameter}:{label}"] = _sha256_file(path)
    return _binding_from_file_digests(contract, file_digests)


def _resolved_config_path(value: Any, *, repo_root: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ActionCalibrationError("action calibration path must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def validate_action_calibration_config(
    contract: ActionCalibrationContract,
    policy_params: Mapping[str, Any],
    *,
    repo_root: Path | str | None = None,
) -> ActionCalibrationBinding:
    """Verify composed policy parameters select exactly the reviewed calibration."""
    if not isinstance(policy_params, Mapping):
        raise ActionCalibrationError("policy.params must be an object for action calibration")
    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    reviewed_files = reviewed_calibration_files(root)
    for parameter, label in contract.file_parameters.items():
        expected_path = reviewed_files.get(label)
        if expected_path is None:
            raise ActionCalibrationError(
                f"action calibration {contract.calibration_id!r} references unknown "
                f"reviewed label {label!r}"
            )
        actual_path = _resolved_config_path(policy_params.get(parameter), repo_root=root)
        if actual_path != expected_path.resolve():
            raise ActionCalibrationError(
                f"policy.params.{parameter} must resolve to reviewed {label!r}, "
                f"got {actual_path}"
            )
    for parameter in contract.absent_parameters:
        if policy_params.get(parameter) not in (None, ""):
            raise ActionCalibrationError(
                f"policy.params.{parameter} must be absent for action calibration "
                f"{contract.calibration_id!r}"
            )
    for parameter, expected in contract.required_policy_values.items():
        if policy_params.get(parameter) != expected:
            raise ActionCalibrationError(
                f"policy.params.{parameter} must equal {expected!r} for action "
                f"calibration {contract.calibration_id!r}"
            )
    return reviewed_action_calibration(contract, repo_root=root)


def validate_action_calibration_binding(
    binding_id: Any,
    binding_digest: Any,
    expected: ActionCalibrationBinding,
) -> None:
    """Reject an evaluator whose local action conversion differs from its binding."""
    calibration_id = _identifier(
        binding_id, field_name="binding action_calibration_id"
    )
    digest = str(binding_digest or "").strip().lower()
    if _SHA256_RE.fullmatch(digest) is None:
        raise ActionCalibrationError(
            "binding action_calibration_digest must be a SHA-256 digest"
        )
    if calibration_id != expected.calibration_id or digest != expected.digest:
        raise ActionCalibrationError(
            "evaluator action calibration does not match the trusted observation binding"
        )


def snapshot_action_calibrations(
    pairs: Sequence[tuple[ActionCalibrationContract, ActionCalibrationBinding]],
    destination: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> Path:
    """Copy verified calibration bytes into one immutable comparison job directory."""
    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    snapshot_root = Path(destination).expanduser().resolve()
    try:
        snapshot_root.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise ActionCalibrationError(
            f"could not create action calibration snapshot {snapshot_root}"
        ) from exc

    reviewed_files = reviewed_calibration_files(root)
    copied_labels: dict[str, str] = {}
    for contract, binding in pairs:
        expected = _binding_from_file_digests(contract, binding.file_digests)
        validate_action_calibration_binding(
            binding.calibration_id,
            binding.digest,
            expected,
        )
        for parameter, label in contract.file_parameters.items():
            digest_key = f"{parameter}:{label}"
            expected_digest = expected.file_digests[digest_key]
            previous = copied_labels.get(label)
            if previous is not None and previous != expected_digest:
                raise ActionCalibrationError(
                    f"action calibration label {label!r} has conflicting digests"
                )
            source = reviewed_files.get(label)
            if source is None or not source.is_file():
                raise ActionCalibrationError(
                    f"reviewed action calibration file for {label!r} is missing"
                )
            if _sha256_file(source) != expected_digest:
                raise ActionCalibrationError(
                    f"reviewed action calibration {label!r} changed after preflight"
                )
            destination_path = snapshot_root / label
            if previous is None:
                shutil.copyfile(source, destination_path)
                copied_labels[label] = expected_digest
    for contract, binding in pairs:
        action_calibration_snapshot_paths(contract, binding, snapshot_root)
    return snapshot_root


def action_calibration_snapshot_paths(
    contract: ActionCalibrationContract,
    binding: ActionCalibrationBinding,
    snapshot_root: Path | str,
) -> dict[str, str]:
    """Verify an action-calibration snapshot and return absolute policy overrides."""
    expected = _binding_from_file_digests(contract, binding.file_digests)
    validate_action_calibration_binding(
        binding.calibration_id,
        binding.digest,
        expected,
    )
    root = Path(snapshot_root).expanduser().resolve()
    paths: dict[str, str] = {}
    for parameter, label in contract.file_parameters.items():
        path = root / label
        expected_digest = expected.file_digests[f"{parameter}:{label}"]
        if not path.is_file() or _sha256_file(path) != expected_digest:
            raise ActionCalibrationError(
                f"action calibration snapshot for {label!r} does not match its binding"
            )
        paths[parameter] = str(path)
    return paths


def _reviewed_action_calibration_paths(
    contract: ActionCalibrationContract,
    binding: ActionCalibrationBinding,
    *,
    repo_root: Path | str | None = None,
) -> dict[str, str]:
    """Return verified repository calibration paths for an unbound evaluator."""
    expected = _binding_from_file_digests(contract, binding.file_digests)
    validate_action_calibration_binding(
        binding.calibration_id,
        binding.digest,
        expected,
    )
    reviewed_files = reviewed_calibration_files(repo_root)
    paths: dict[str, str] = {}
    for parameter, label in contract.file_parameters.items():
        path = reviewed_files[label].resolve()
        expected_digest = expected.file_digests[f"{parameter}:{label}"]
        if not path.is_file() or _sha256_file(path) != expected_digest:
            raise ActionCalibrationError(
                f"reviewed action calibration {label!r} changed during evaluation setup"
            )
        paths[parameter] = str(path)
    return paths


def prepare_action_calibration(
    contract: ActionCalibrationContract,
    policy_params: Mapping[str, Any],
    *,
    binding_id: Any | None = None,
    binding_file_digests: Mapping[str, Any] | None = None,
    binding_digest: Any | None = None,
    snapshot_root: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> tuple[ActionCalibrationBinding, dict[str, str]]:
    """Validate config semantics and, for bound jobs, select frozen file paths."""
    expected = validate_action_calibration_config(
        contract,
        policy_params,
        repo_root=repo_root,
    )
    supplied = (binding_id, binding_file_digests, binding_digest, snapshot_root)
    if all(value is None for value in supplied):
        return expected, _reviewed_action_calibration_paths(
            contract,
            expected,
            repo_root=repo_root,
        )
    if any(value is None for value in supplied):
        raise ActionCalibrationError(
            "trusted action calibration requires binding fields and a snapshot directory"
        )
    assert binding_file_digests is not None
    supplied_binding = _binding_from_file_digests(contract, binding_file_digests)
    validate_action_calibration_binding(binding_id, binding_digest, supplied_binding)
    validate_action_calibration_binding(
        supplied_binding.calibration_id,
        supplied_binding.digest,
        expected,
    )
    assert snapshot_root is not None
    return expected, action_calibration_snapshot_paths(
        contract,
        supplied_binding,
        snapshot_root,
    )


__all__ = [
    "ACTION_CALIBRATION_SCHEMA",
    "ActionCalibrationBinding",
    "ActionCalibrationContract",
    "ActionCalibrationError",
    "action_calibration_snapshot_paths",
    "prepare_action_calibration",
    "reviewed_action_calibration",
    "snapshot_action_calibrations",
    "validate_action_calibration_binding",
    "validate_action_calibration_config",
]
