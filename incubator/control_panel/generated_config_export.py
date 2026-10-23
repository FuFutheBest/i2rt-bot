"""Safely publish explicit runtime configurations.

The Control Panel keeps transient hardware bindings in memory. This module
turns an already-resolved, effective configuration into an explicit Hydra
launch file. Generated exports are isolated from the selected source preset;
the separate source-persistence entry point is always explicit. Generated files
use the same root-relative namespace as `device_starter.py`
(``configs/generated_*.yaml``).
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .gripper_contract import rizon_gripper_side_count

GENERATED_CONFIG_SCHEMA_VERSION = 1
_MAX_CONFIG_BYTES = 8 * 1024 * 1024
_MAX_CALIBRATION_BYTES = 4 * 1024 * 1024
_MAX_VALUE_DEPTH = 64
_MAX_VALUE_ITEMS = 100_000
_SAFE_GENERATED_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_SAFE_SOURCE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_SAFE_REVISION_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_POLICY_INFER_SOURCE_NAMESPACES = frozenset({"fastwam", "lingbotva", "openpi"})
_DELETE_YAML_NODE = object()
_TUNING_LIMITS = {
    "policy": {
        "fps": (0.1, 30.0, False),
        "control_fps": (1.0, 120.0, False),
        "chunk_length": (1, 512, True),
    },
}


class _IndentedSafeDumper(yaml.SafeDumper):
    """Use the two-space sequence indentation used by repository configs."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        super().increase_indent(flow=flow, indentless=False)


class GeneratedConfigExportError(ValueError):
    """Raised when an export request or target cannot be made safe."""


class _GeneratedConfigPublishedError(GeneratedConfigExportError):
    """The config filename became visible before its final directory sync failed."""


def _calibration_side_enabled(gripper: Mapping[str, Any], index: int) -> bool:
    """Resolve a per-side enable flag without dropping legacy scalar configs."""
    value: Any = gripper.get("enabled", True)
    if isinstance(value, (list, tuple)):
        value = value[index] if index < len(value) else False
    if isinstance(value, str):
        return value.strip().casefold() not in {"", "0", "false", "no", "off"}
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            return value != 0 and math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    return True


def _is_rizon_robot_class(value: Any) -> bool:
    return str(value or "").casefold().startswith("rizonrobot")


def _normalize_legacy_gripper_transport(config: dict[str, Any]) -> list[str]:
    """Remove the deleted WebSocket transport from an export copy.

    The server normally performs this migration before calling the exporter,
    but the exporter is also a public library boundary.  Keep direct callers
    from publishing a generated preset that still depends on the removed
    daemon, while preserving legitimate Flexiv gripper backends.
    """
    devices = config.get("devices")
    if isinstance(devices, Mapping):
        devices = devices.get("devices", [])
    if not isinstance(devices, list):
        return []
    warnings: list[str] = []
    for device in devices:
        if not isinstance(device, dict) or not _is_rizon_robot_class(device.get("class")):
            continue
        gripper = device.get("gripper")
        if isinstance(gripper, dict):
            backend = str(gripper.get("backend", "")).strip().casefold()
            if backend == "lktech_websocket":
                gripper["backend"] = "lktech"
                gripper.pop("server", None)
                warnings.append("legacy gripper backend lktech_websocket was normalized to direct lktech")
            elif backend in {"", "lktech"} and "server" in gripper:
                if not backend:
                    gripper["backend"] = "lktech"
                gripper.pop("server", None)
                warnings.append("legacy gripper server fields were ignored (default direct lktech)")
            continue
        flat_backend = str(device.get("gripper_backend", "")).strip().casefold()
        if flat_backend == "lktech_websocket":
            device["gripper_backend"] = "lktech"
            device.pop("gripper_server", None)
            warnings.append("legacy flat gripper backend lktech_websocket was normalized to direct lktech")
        elif flat_backend in {"", "lktech"} and "gripper_server" in device:
            # The removed WebSocket daemon used this field.  An empty flat
            # backend historically selected direct LKTech, so make that
            # default explicit while keeping the caller's mapping untouched.
            if not flat_backend:
                device["gripper_backend"] = "lktech"
            device.pop("gripper_server", None)
            warnings.append("legacy gripper server fields were ignored (default direct lktech)")
    return warnings


@dataclass
class _CalibrationSnapshotRoot:
    """Descriptor-anchored generated-calibration directory.

    Generated calibration paths are consumed later relative to the repository.
    Keeping both parent descriptors open lets this exporter reject a path swap
    before it publishes a YAML file that refers to the lexical path.
    """

    display_path: Path
    repository_descriptor: int
    calibration_descriptor: int
    control_panel_descriptor: int
    calibration_identity: tuple[int, int]
    control_panel_identity: tuple[int, int]
    _closed: bool = False

    def assert_still_linked(self) -> None:
        """Reject replacement of calibration/control_panel after the FD was opened."""
        if self._closed:
            raise GeneratedConfigExportError("generated calibration root is closed")
        try:
            calibration_info = os.stat(
                "calibration",
                dir_fd=self.repository_descriptor,
                follow_symlinks=False,
            )
            control_panel_info = os.stat(
                "control_panel",
                dir_fd=self.calibration_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise GeneratedConfigExportError(
                "generated calibration root changed during snapshot export"
            ) from error
        if (
            not stat.S_ISDIR(calibration_info.st_mode)
            or (calibration_info.st_dev, calibration_info.st_ino)
            != self.calibration_identity
            or not stat.S_ISDIR(control_panel_info.st_mode)
            or (control_panel_info.st_dev, control_panel_info.st_ino)
            != self.control_panel_identity
        ):
            raise GeneratedConfigExportError(
                "generated calibration root changed during snapshot export"
            )

    def close(self) -> None:
        """Release retained directory descriptors without masking an export error."""
        if self._closed:
            return
        self._closed = True
        for descriptor in (
            self.control_panel_descriptor,
            self.calibration_descriptor,
            self.repository_descriptor,
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass


@dataclass(frozen=True)
class GeneratedConfigExport:
    """Stable summary returned after an exported config is durably published."""

    config_name: str
    path: Path
    revision: str
    source_config_name: str
    generated_at: str
    content_sha256: str
    backup_path: Path | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation for a future API handler."""
        return {
            "config_name": self.config_name,
            "path": str(self.path),
            "revision": self.revision,
            "source_config_name": self.source_config_name,
            "generated_at": self.generated_at,
            "content_sha256": self.content_sha256,
            "backup_path": str(self.backup_path) if self.backup_path else None,
        }


def export_generated_config(
    effective_config: Mapping[str, Any],
    *,
    config_dir: Path,
    source_config_name: str,
    revision: int | str,
    generated_id: str,
    bindings: Mapping[str, Any] | None = None,
    calibration_artifacts: Mapping[str, Any] | None = None,
    tuning_overrides: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    repository_root: Path | None = None,
) -> GeneratedConfigExport:
    """Write an effective config as a new, isolated shared Hydra launch preset.

    ``effective_config`` must already include the session's selected hardware
    bindings.  ``bindings`` and ``calibration_artifacts`` are persisted only
    as provenance so an operator can audit how the generated file was made.
    ``tuning_overrides`` can change only the restart-aware FPS/chunk allowlist.
    The function never edits ``source_config_name`` or any other source preset.

    The generated Hydra name is ``generated_<generated_id>``. Existing output
    is replaced atomically only when ``overwrite`` is true.
    """
    if not isinstance(overwrite, bool):
        raise GeneratedConfigExportError("overwrite must be a boolean")

    normalized_generated_id = _normalize_generated_id(generated_id)
    normalized_source_name = _normalize_source_config_name(source_config_name)
    normalized_revision = _normalize_revision(revision)
    normalized_effective = _copy_safe_mapping(effective_config, field="effective_config")
    legacy_transport_warnings = _normalize_legacy_gripper_transport(normalized_effective)
    if "control_panel_export" in normalized_effective:
        raise GeneratedConfigExportError(
            "effective_config reserves control_panel_export for generated provenance"
        )

    target_root = _generated_root(config_dir)
    target = target_root / f"generated_{normalized_generated_id}.yaml"
    _assert_safe_target(target, root=target_root)
    del repository_root  # Retained for compatibility with existing direct callers.
    normalized_bindings = _copy_safe_mapping(bindings or {}, field="bindings")
    normalized_calibrations = _copy_safe_mapping(
        calibration_artifacts or {}, field="calibration_artifacts"
    )
    normalized_tuning = _normalize_tuning_overrides(tuning_overrides or {})
    _apply_tuning_overrides(normalized_effective, normalized_tuning)
    generated_at = datetime.now(timezone.utc).isoformat()
    generated_name = f"generated_{normalized_generated_id}"
    provenance = {
        "schema_version": GENERATED_CONFIG_SCHEMA_VERSION,
        "generated_at": generated_at,
        "source_config_name": normalized_source_name,
        "source_revision": normalized_revision,
        "bindings": normalized_bindings,
        "calibration_artifacts": normalized_calibrations,
        "tuning_overrides": normalized_tuning,
    }
    if legacy_transport_warnings:
        provenance["legacy_transport_migrations"] = legacy_transport_warnings
    normalized_effective["control_panel_export"] = provenance
    payload = _serialize_yaml(
        _generated_payload_with_source_layout(
            normalized_effective,
            config_dir=target_root,
            source_config_name=normalized_source_name,
        )
    )
    if len(payload) > _MAX_CONFIG_BYTES:
        raise GeneratedConfigExportError(
            f"generated configuration exceeds {_MAX_CONFIG_BYTES} bytes"
        )

    _publish_atomically(target, payload, overwrite=overwrite)
    return GeneratedConfigExport(
        config_name=generated_name,
        path=target,
        revision=normalized_revision,
        source_config_name=normalized_source_name,
        generated_at=generated_at,
        content_sha256=hashlib.sha256(payload).hexdigest(),
        backup_path=None,
    )


def persist_source_config(
    effective_config: Mapping[str, Any],
    *,
    config_dir: Path,
    source_config_name: str,
    revision: int | str,
    tuning_overrides: Mapping[str, Any] | None = None,
) -> GeneratedConfigExport:
    """Persist changed panel leaves to the selected launch config and groups.

    This is intentionally an explicit operation.  The selected source file and
    any referenced Hydra group files are patched at their existing YAML nodes,
    preserving comments, ordering, and interpolation. Every changed file is
    replaced atomically without creating sidecar backups, while the caller's
    mapping is never mutated. Panel provenance is not embedded in source presets.
    """
    normalized_source_name = _normalize_source_config_name(source_config_name)
    normalized_revision = _normalize_revision(revision)
    normalized_effective = _copy_safe_mapping(effective_config, field="effective_config")
    normalized_effective.pop("control_panel_export", None)
    normalized_tuning = _normalize_tuning_overrides(tuning_overrides or {})
    _apply_tuning_overrides(normalized_effective, normalized_tuning)

    source_root = _generated_root(config_dir)
    relative = Path(*normalized_source_name.split("/"))
    yaml_target = source_root / relative.with_suffix(".yaml")
    yml_target = source_root / relative.with_suffix(".yml")
    target = yaml_target if yaml_target.exists() or not yml_target.exists() else yml_target
    _assert_safe_source_target(target, root=source_root)
    existing = _read_existing_regular_file(target)
    if existing is None:
        raise GeneratedConfigExportError(
            f"source config does not exist: {normalized_source_name}"
        )

    source_document = _load_yaml_document(existing, target)
    updates = _source_updates(
        normalized_effective,
        source_document,
        normalized_tuning,
    )

    # Keep the source file byte-for-byte stable when this session did not
    # change a value.  This also avoids creating a misleading backup for a
    # no-op save.
    source_payload = existing
    if updates.get("source"):
        source_payload = _patch_yaml_bytes(existing, updates["source"], target)
        if len(source_payload) > _MAX_CONFIG_BYTES:
            raise GeneratedConfigExportError(
                f"source configuration exceeds {_MAX_CONFIG_BYTES} bytes"
            )

    def check_source_target(candidate: Path, *, root: Path) -> None:
        del root
        _assert_safe_source_target(candidate, root=source_root)

    backup_path = None
    if source_payload != existing:
        backup_path = _publish_atomically(
            target,
            source_payload,
            overwrite=True,
            safe_target_check=check_source_target,
        )
    for group_relative, group_updates in updates.get("groups", ()):
        group_target = source_root / group_relative
        _assert_safe_source_target(group_target, root=source_root)
        group_existing = _read_existing_regular_file(group_target)
        if group_existing is None:
            raise GeneratedConfigExportError(
                f"config group does not exist: {group_target}"
            )
        group_document = _load_yaml_document(group_existing, group_target)
        group_name = group_target.parent.name
        effective_group_updates: dict[tuple[Any, ...], Any] = {}
        for update_path, value in group_updates.items():
            if update_path and update_path[0] in group_document:
                effective_group_updates[update_path] = value
            elif group_name in group_document and isinstance(group_document[group_name], Mapping):
                effective_group_updates[(group_name, *update_path)] = value
            else:
                effective_group_updates[update_path] = value
        group_payload = _patch_yaml_bytes(
            group_existing,
            effective_group_updates,
            group_target,
        )
        if group_payload == group_existing:
            continue
        _publish_atomically(
            group_target,
            group_payload,
            overwrite=True,
            safe_target_check=lambda candidate, *, root: _assert_safe_source_target(
                candidate, root=source_root
            ),
        )
    payload = source_payload
    return GeneratedConfigExport(
        config_name=normalized_source_name,
        path=target.resolve(strict=False),
        revision=normalized_revision,
        source_config_name=normalized_source_name,
        generated_at=datetime.now(timezone.utc).isoformat(),
        content_sha256=hashlib.sha256(payload).hexdigest(),
        backup_path=backup_path,
    )


@dataclass(frozen=True)
class _CalibrationSnapshotPlan:
    root: _CalibrationSnapshotRoot
    destination_name: str
    files: tuple[tuple[str, bytes], ...]
    manifest: dict[str, dict[str, Any]]

    @property
    def destination(self) -> Path:
        return self.root.display_path / self.destination_name


def _freeze_gripper_calibration_files(
    effective_config: dict[str, Any],
    *,
    repository_root: Path,
    generated_id: str,
) -> _CalibrationSnapshotPlan | None:
    """Copy configured gripper calibration inputs into an immutable run snapshot."""

    devices = effective_config.get("devices")
    if isinstance(devices, Mapping):
        devices = devices.get("devices")
    if not isinstance(devices, list):
        return None

    repository_root = Path(os.path.abspath(repository_root))
    calibration_root = repository_root / "calibration"
    entries: list[tuple[str, bytes]] = []
    manifest: dict[str, dict[str, Any]] = {}
    frozen_paths: list[tuple[dict[str, Any], list[str], bool]] = []
    for device_index, device in enumerate(devices):
        if not isinstance(device, dict):
            continue
        if not _is_rizon_robot_class(device.get("class")):
            continue
        gripper = device.get("gripper")
        if not isinstance(gripper, dict):
            # Keep the standalone exporter compatible with the older explicit
            # flat LKTech form.  The nested form is canonical for generated
            # configs and takes precedence at runtime.
            flat_backend = str(device.get("gripper_backend") or "").strip().casefold()
            if flat_backend not in {"lktech", ""}:
                continue
            if flat_backend == "" and "gripper_server" not in device:
                continue
            gripper = {
                "backend": "lktech",
                "enabled": device.get("utilize_gripper", False),
                "port": device.get("gripper_port"),
                "motor_id": device.get("gripper_motor_id", 1),
                "calibration_files": device.get("gripper_calibration_files"),
            }
            device["gripper"] = gripper
        if not isinstance(gripper, dict):
            continue
        raw_files = gripper.get("calibration_files")
        if raw_files is None:
            continue
        side_count = rizon_gripper_side_count(device, gripper)
        scalar_files = isinstance(raw_files, str)
        if scalar_files:
            raw_file_values = [raw_files]
        elif isinstance(raw_files, list):
            raw_file_values = list(raw_files)
        else:
            raw_file_values = []
        if side_count == 2 and len(raw_file_values) != 2:
            raise GeneratedConfigExportError(
                "Rizon gripper.calibration_files must contain left and right JSON paths"
            )
        if side_count == 1 and len(raw_file_values) not in {1, 2}:
            raise GeneratedConfigExportError(
                "single-arm Rizon gripper.calibration_files must contain one JSON path"
            )
        if (
            side_count == 1
            and len(raw_file_values) == 2
            and isinstance(gripper.get("enabled"), (list, tuple))
            and len(gripper["enabled"]) > 1
            and _calibration_side_enabled(gripper, 1)
        ):
            raise GeneratedConfigExportError(
                "single-arm Rizon gripper cannot enable a second calibration slot"
            )

        replacement: list[str] = []
        for side_index, raw_path in enumerate(raw_file_values):
            if side_index >= side_count:
                # Some historical single-arm presets retained a second empty
                # slot. Preserve it verbatim, but never snapshot or activate it.
                replacement.append(str(raw_path) if raw_path is not None else "")
                continue
            side = ("left", "right")[side_index]
            if not _calibration_side_enabled(gripper, side_index):
                replacement.append(str(raw_path) if raw_path is not None else "")
                continue
            source_path, source_bytes = _read_safe_calibration_source(
                raw_path,
                repository_root=repository_root,
                calibration_root=calibration_root,
            )
            relative_name = f"rizon_{device_index}_{side}.json"
            relative_snapshot = (
                Path("calibration") / "control_panel" / "{snapshot}" / relative_name
            )
            replacement.append(str(relative_snapshot))
            source_relative = source_path.relative_to(repository_root).as_posix()
            digest = hashlib.sha256(source_bytes).hexdigest()
            manifest[f"rizon_{device_index}_{side}"] = {
                "source_path": source_relative,
                "snapshot_path": str(relative_snapshot),
                "sha256": digest,
                "bytes": len(source_bytes),
            }
            entries.append((relative_name, source_bytes))
        frozen_paths.append((gripper, replacement, scalar_files))

    if not entries:
        return None

    snapshot_root = _generated_calibration_root(repository_root)
    try:
        snapshot_name = f"generated_{generated_id}_{uuid.uuid4().hex}"
        snapshot_root.assert_still_linked()
        if _directory_entry_exists(snapshot_root.control_panel_descriptor, snapshot_name):
            raise GeneratedConfigExportError(
                f"unsafe calibration snapshot target: {snapshot_root.display_path / snapshot_name}"
            )
        for gripper, replacement, scalar_files in frozen_paths:
            formatted = [path.format(snapshot=snapshot_name) for path in replacement]
            gripper["calibration_files"] = formatted[0] if scalar_files else formatted
        for item in manifest.values():
            item["snapshot_path"] = str(item["snapshot_path"]).format(snapshot=snapshot_name)
        return _CalibrationSnapshotPlan(
            root=snapshot_root,
            destination_name=snapshot_name,
            files=tuple(entries),
            manifest=manifest,
        )
    except BaseException:
        snapshot_root.close()
        raise


def _directory_open_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        raise GeneratedConfigExportError(
            "safe calibration export requires O_NOFOLLOW and O_DIRECTORY support"
        )
    return os.O_RDONLY | nofollow | directory_flag


def _open_directory(
    path: Path | str,
    *,
    context: str,
    dir_fd: int | None = None,
) -> int:
    try:
        if dir_fd is None:
            descriptor = os.open(os.fspath(path), _directory_open_flags())
        else:
            descriptor = os.open(
                os.fspath(path),
                _directory_open_flags(),
                dir_fd=dir_fd,
            )
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise GeneratedConfigExportError(f"{context} must not be a symbolic link") from error
        raise GeneratedConfigExportError(f"could not safely open {context}") from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise GeneratedConfigExportError(f"{context} must be a directory")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _directory_entry_exists(directory_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise GeneratedConfigExportError(
            f"could not inspect generated calibration entry: {name}"
        ) from error
    return True


def _generated_calibration_root(repository_root: Path) -> _CalibrationSnapshotRoot:
    """Open calibration/control_panel without path-based publish operations."""
    repository_root = Path(os.path.abspath(repository_root))
    display_path = repository_root / "calibration" / "control_panel"
    repository_descriptor = _open_directory(
        repository_root,
        context=f"repository root: {repository_root}",
    )
    calibration_descriptor: int | None = None
    control_panel_descriptor: int | None = None
    try:
        calibration_descriptor = _open_directory(
            "calibration",
            context=f"calibration root: {repository_root / 'calibration'}",
            dir_fd=repository_descriptor,
        )
        try:
            os.mkdir("control_panel", mode=0o700, dir_fd=calibration_descriptor)
        except FileExistsError:
            pass
        control_panel_descriptor = _open_directory(
            "control_panel",
            context=f"generated calibration root: {display_path}",
            dir_fd=calibration_descriptor,
        )
        calibration_info = os.fstat(calibration_descriptor)
        control_panel_info = os.fstat(control_panel_descriptor)
        root = _CalibrationSnapshotRoot(
            display_path=display_path,
            repository_descriptor=repository_descriptor,
            calibration_descriptor=calibration_descriptor,
            control_panel_descriptor=control_panel_descriptor,
            calibration_identity=(calibration_info.st_dev, calibration_info.st_ino),
            control_panel_identity=(control_panel_info.st_dev, control_panel_info.st_ino),
        )
        root.assert_still_linked()
        return root
    except BaseException:
        if control_panel_descriptor is not None:
            try:
                os.close(control_panel_descriptor)
            except OSError:
                pass
        if calibration_descriptor is not None:
            try:
                os.close(calibration_descriptor)
            except OSError:
                pass
        try:
            os.close(repository_descriptor)
        except OSError:
            pass
        raise


def _read_safe_calibration_source(
    raw_path: Any,
    *,
    repository_root: Path,
    calibration_root: Path,
) -> tuple[Path, bytes]:
    """Read calibration JSON through a no-symlink descriptor walk.

    A lexical root check alone is not sufficient: another process could replace
    an intermediate directory after validation. Open every path component below
    the calibration root relative to an already-open directory descriptor, with
    O_NOFOLLOW on each component, so copied bytes came from the reviewed tree.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise GeneratedConfigExportError(
            "gripper calibration_files must contain non-empty JSON paths"
        )

    repository_root = Path(os.path.abspath(repository_root))
    calibration_root = Path(os.path.abspath(calibration_root))
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(calibration_root)
    except ValueError as error:
        raise GeneratedConfigExportError(
            "gripper calibration file must remain under repository calibration/"
        ) from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise GeneratedConfigExportError(
            "gripper calibration file must name a regular JSON file below calibration/"
        )

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        raise GeneratedConfigExportError(
            "safe calibration export requires O_NOFOLLOW and O_DIRECTORY support"
        )

    directory_flags = os.O_RDONLY | nofollow | directory_flag
    try:
        directory_descriptor = os.open(calibration_root, directory_flags)
    except OSError as error:
        raise GeneratedConfigExportError(
            f"calibration root must be an existing non-symlink directory: {calibration_root}"
        ) from error

    descriptor: int | None = None
    try:
        for component in relative.parts[:-1]:
            try:
                next_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=directory_descriptor,
                )
            except OSError as error:
                if error.errno == errno.ELOOP:
                    raise GeneratedConfigExportError(
                        "gripper calibration path must not contain symbolic links"
                    ) from error
                if error.errno == errno.ENOENT:
                    raise GeneratedConfigExportError(
                        f"gripper calibration file does not exist: {candidate}"
                    ) from error
                raise GeneratedConfigExportError(
                    f"could not safely open gripper calibration directory: {component}"
                ) from error
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor

        try:
            descriptor = os.open(
                relative.parts[-1],
                os.O_RDONLY | nofollow,
                dir_fd=directory_descriptor,
            )
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise GeneratedConfigExportError(
                    "gripper calibration path must not contain symbolic links"
                ) from error
            if error.errno == errno.ENOENT:
                raise GeneratedConfigExportError(
                    f"gripper calibration file does not exist: {candidate}"
                ) from error
            raise GeneratedConfigExportError(
                f"could not read gripper calibration file: {candidate}"
            ) from error

        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise GeneratedConfigExportError(
                f"gripper calibration file is not regular: {candidate}"
            )
        if info.st_size > _MAX_CALIBRATION_BYTES:
            raise GeneratedConfigExportError(
                f"gripper calibration file is too large: {candidate}"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_CALIBRATION_BYTES:
                raise GeneratedConfigExportError(
                    f"gripper calibration file is too large: {candidate}"
                )
            chunks.append(chunk)
        content = b"".join(chunks)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_descriptor)

    try:
        decoded = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GeneratedConfigExportError(
            f"gripper calibration file is not valid JSON: {candidate}"
        ) from error
    if not isinstance(decoded, Mapping):
        raise GeneratedConfigExportError(
            f"gripper calibration file must contain a JSON object: {candidate}"
        )
    return candidate, content


def _snapshot_entry_name(name: str) -> None:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise GeneratedConfigExportError("calibration snapshot target escaped its root")


def _write_private_file_at(directory_descriptor: int, name: str, content: bytes) -> None:
    _snapshot_entry_name(name)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            os.unlink(name, dir_fd=directory_descriptor)
        except OSError:
            pass
        raise


def _remove_private_snapshot_directory(
    root: _CalibrationSnapshotRoot,
    directory_name: str,
    files: Sequence[tuple[str, bytes]],
) -> None:
    """Best-effort cleanup through the retained directory FD only."""
    try:
        _snapshot_entry_name(directory_name)
        directory_descriptor = _open_directory(
            directory_name,
            context="generated calibration staging directory",
            dir_fd=root.control_panel_descriptor,
        )
    except GeneratedConfigExportError:
        return

    remove_directory = True
    try:
        for name, _ in files:
            _snapshot_entry_name(name)
            try:
                info = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            except OSError:
                remove_directory = False
                break
            if not stat.S_ISREG(info.st_mode):
                remove_directory = False
                break
            try:
                os.unlink(name, dir_fd=directory_descriptor)
            except OSError:
                remove_directory = False
                break
    finally:
        try:
            os.close(directory_descriptor)
        except OSError:
            pass

    if not remove_directory:
        return
    try:
        os.rmdir(directory_name, dir_fd=root.control_panel_descriptor)
        os.fsync(root.control_panel_descriptor)
    except OSError:
        pass


def _publish_calibration_snapshot(plan: _CalibrationSnapshotPlan) -> Path:
    root = plan.root
    root.assert_still_linked()
    staging_name = f".calibration-{uuid.uuid4().hex}"
    staging_descriptor: int | None = None
    staging_created = False
    renamed = False
    try:
        if _directory_entry_exists(root.control_panel_descriptor, plan.destination_name):
            raise GeneratedConfigExportError(
                f"unsafe calibration snapshot target: {plan.destination}"
            )
        os.mkdir(staging_name, mode=0o700, dir_fd=root.control_panel_descriptor)
        staging_created = True
        staging_descriptor = _open_directory(
            staging_name,
            context="generated calibration staging directory",
            dir_fd=root.control_panel_descriptor,
        )
        os.fchmod(staging_descriptor, 0o700)
        for name, content in plan.files:
            _write_private_file_at(staging_descriptor, name, content)
        os.fsync(staging_descriptor)
        os.close(staging_descriptor)
        staging_descriptor = None

        root.assert_still_linked()
        if _directory_entry_exists(root.control_panel_descriptor, plan.destination_name):
            raise GeneratedConfigExportError(
                f"unsafe calibration snapshot target: {plan.destination}"
            )
        os.rename(
            staging_name,
            plan.destination_name,
            src_dir_fd=root.control_panel_descriptor,
            dst_dir_fd=root.control_panel_descriptor,
        )
        renamed = True
        os.fsync(root.control_panel_descriptor)
        root.assert_still_linked()
        return plan.destination
    except BaseException:
        if staging_descriptor is not None:
            try:
                os.close(staging_descriptor)
            except OSError:
                pass
        # A signal after rename but before the local flag is set leaves an
        # orphaned private snapshot. That is safer than deleting an object that
        # may shortly be referenced by a YAML file.
        if staging_created and not renamed:
            _remove_private_snapshot_directory(root, staging_name, plan.files)
        raise


def _remove_calibration_snapshot(plan: _CalibrationSnapshotPlan) -> None:
    _remove_private_snapshot_directory(plan.root, plan.destination_name, plan.files)


def _normalize_generated_id(value: Any) -> str:
    if not isinstance(value, str):
        raise GeneratedConfigExportError(
            "generated_id must contain only letters, digits, '.', '_' or '-'"
        )
    normalized = value.strip()
    if (
        _SAFE_GENERATED_ID_RE.fullmatch(normalized) is None
        or normalized.endswith((".yaml", ".yml"))
    ):
        raise GeneratedConfigExportError(
            "generated_id must contain only letters, digits, '.', '_' or '-'"
        )
    return normalized


def _normalize_source_config_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GeneratedConfigExportError("source_config_name is required")
    normalized = value.strip()
    if normalized.endswith(".yaml"):
        normalized = normalized[:-5]
    elif normalized.endswith(".yml"):
        normalized = normalized[:-4]
    parts = normalized.split("/")
    valid_namespace = (
        len(parts) == 1
        or (
            len(parts) == 3
            and parts[0] == "policy_infer"
            and parts[1] in _POLICY_INFER_SOURCE_NAMESPACES
        )
    )
    if (
        not parts
        or any(part in {"", ".", ".."} for part in parts)
        or any(_SAFE_SOURCE_COMPONENT_RE.fullmatch(part) is None for part in parts)
        or not valid_namespace
    ):
        raise GeneratedConfigExportError(
            "source_config_name must be a relative Hydra config name"
        )
    return normalized


def _normalize_revision(value: Any) -> str:
    if isinstance(value, bool):
        raise GeneratedConfigExportError("revision must be a non-negative integer or identifier")
    if isinstance(value, int):
        if value < 0:
            raise GeneratedConfigExportError("revision must be non-negative")
        return str(value)
    if isinstance(value, str) and _SAFE_REVISION_RE.fullmatch(value.strip()):
        return value.strip()
    raise GeneratedConfigExportError("revision must be a non-negative integer or identifier")


def _copy_safe_mapping(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GeneratedConfigExportError(f"{field} must be a mapping")
    item_count = [0]
    return _copy_safe_value(value, field=field, depth=0, active=set(), item_count=item_count)


def _normalize_tuning_overrides(value: Mapping[str, Any]) -> dict[str, dict[str, float | int]]:
    safe_value = _copy_safe_mapping(value, field="tuning_overrides")
    normalized: dict[str, dict[str, float | int]] = {}
    for section, fields in safe_value.items():
        allowed_fields = _TUNING_LIMITS.get(section)
        if allowed_fields is None:
            raise GeneratedConfigExportError(f"tuning_overrides.{section} is not allowed")
        if not isinstance(fields, Mapping):
            raise GeneratedConfigExportError(f"tuning_overrides.{section} must be a mapping")
        normalized_section: dict[str, float | int] = {}
        for field, raw_value in fields.items():
            limits = allowed_fields.get(field)
            if limits is None:
                raise GeneratedConfigExportError(
                    f"tuning_overrides.{section}.{field} is not allowed"
                )
            minimum, maximum, integral = limits
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise GeneratedConfigExportError(
                    f"tuning_overrides.{section}.{field} must be numeric"
                )
            numeric_value = float(raw_value)
            if not math.isfinite(numeric_value) or not minimum <= numeric_value <= maximum:
                raise GeneratedConfigExportError(
                    f"tuning_overrides.{section}.{field} must be between {minimum} and {maximum}"
                )
            if integral:
                if int(numeric_value) != numeric_value:
                    raise GeneratedConfigExportError(
                        f"tuning_overrides.{section}.{field} must be an integer"
                    )
                normalized_section[field] = int(numeric_value)
            else:
                normalized_section[field] = numeric_value
        normalized[section] = normalized_section
    return normalized


def _apply_tuning_overrides(
    effective_config: dict[str, Any], tuning_overrides: Mapping[str, Mapping[str, float | int]]
) -> None:
    for section, fields in tuning_overrides.items():
        existing = effective_config.get(section)
        if not isinstance(existing, dict):
            raise GeneratedConfigExportError(
                f"effective_config.{section} must be a mapping to apply tuning overrides"
            )
        existing.update(fields)


def _copy_safe_value(
    value: Any,
    *,
    field: str,
    depth: int,
    active: set[int],
    item_count: list[int],
) -> Any:
    if depth > _MAX_VALUE_DEPTH:
        raise GeneratedConfigExportError(f"{field} exceeds maximum nesting depth")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GeneratedConfigExportError(f"{field} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise GeneratedConfigExportError(f"{field} contains a recursive mapping")
        active.add(marker)
        copied: dict[str, Any] = {}
        try:
            for key, child in value.items():
                _increment_item_count(field, item_count)
                if not isinstance(key, str) or not key:
                    raise GeneratedConfigExportError(f"{field} mapping keys must be non-empty strings")
                copied[key] = _copy_safe_value(
                    child,
                    field=f"{field}.{key}",
                    depth=depth + 1,
                    active=active,
                    item_count=item_count,
                )
        finally:
            active.remove(marker)
        return copied
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        marker = id(value)
        if marker in active:
            raise GeneratedConfigExportError(f"{field} contains a recursive sequence")
        active.add(marker)
        copied_items: list[Any] = []
        try:
            for index, child in enumerate(value):
                _increment_item_count(field, item_count)
                copied_items.append(
                    _copy_safe_value(
                        child,
                        field=f"{field}[{index}]",
                        depth=depth + 1,
                        active=active,
                        item_count=item_count,
                    )
                )
        finally:
            active.remove(marker)
        return copied_items
    raise GeneratedConfigExportError(
        f"{field} contains unsupported value type {type(value).__name__}"
    )


def _increment_item_count(field: str, item_count: list[int]) -> None:
    item_count[0] += 1
    if item_count[0] > _MAX_VALUE_ITEMS:
        raise GeneratedConfigExportError(f"{field} exceeds maximum item count")


def _serialize_yaml(
    payload: Mapping[str, Any],
    *,
    header: str = (
        "# @package _global_\n"
        "# Generated by the SuperInference Control Panel. Do not edit source presets.\n"
    ),
) -> bytes:
    try:
        body = yaml.dump(
            dict(payload),
            allow_unicode=False,
            default_flow_style=False,
            sort_keys=False,
            Dumper=_IndentedSafeDumper,
        )
    except yaml.YAMLError as error:
        raise GeneratedConfigExportError(f"could not serialize generated configuration: {error}") from error
    return header.encode("utf-8") + body.encode("utf-8")


def _source_config_target(config_root: Path, source_config_name: str) -> Path:
    """Resolve a normalized launch name without following a source symlink."""
    relative = Path(*source_config_name.split("/"))
    yaml_target = config_root / relative.with_suffix(".yaml")
    yml_target = config_root / relative.with_suffix(".yml")
    return yaml_target if yaml_target.exists() or not yml_target.exists() else yml_target


def _load_yaml_document(content: bytes, path: Path) -> Mapping[str, Any]:
    try:
        loaded = yaml.safe_load(content.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as error:
        raise GeneratedConfigExportError(f"could not parse YAML config: {path}") from error
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise GeneratedConfigExportError(f"YAML config root must be a mapping: {path}")
    return loaded


def _generated_payload_with_source_layout(
    effective: Mapping[str, Any],
    *,
    config_dir: Path,
    source_config_name: str,
) -> dict[str, Any]:
    """Retain the source defaults and top-level ordering in a generated file."""
    source = _source_config_target(config_dir, source_config_name)
    existing = _read_existing_regular_file(source)
    if existing is None:
        return dict(effective)
    try:
        source_document = _load_yaml_document(existing, source)
    except GeneratedConfigExportError:
        return dict(effective)
    ordered: dict[str, Any] = {}
    # Keeping defaults makes generated files immediately recognizable and
    # preserves the same Hydra composition contract as the reference preset.
    if "defaults" in source_document:
        ordered["defaults"] = source_document["defaults"]
    default_group_names: set[str] = set()
    raw_defaults = source_document.get("defaults")
    if isinstance(raw_defaults, list):
        for item in raw_defaults:
            if isinstance(item, Mapping):
                for key in item:
                    group_name = str(key).lstrip("/").split("@", maxsplit=1)[0]
                    if group_name == "devices/infer":
                        group_name = "devices"
                    if group_name != "_self_":
                        default_group_names.add(group_name)
    for key in source_document:
        if key != "defaults" and key in effective:
            name = str(key)
            # Hydra group files are packaged below their group name.  Preserve
            # that shape when overriding a resolved value, otherwise a list
            # such as ``devices`` conflicts with the group's mapping node.
            ordered[name] = (
                {name: effective[key]}
                if name in default_group_names
                else effective[key]
            )
    for key, value in effective.items():
        if key not in ordered:
            name = str(key)
            ordered[name] = (
                {name: value}
                if name in default_group_names
                else value
            )
    return ordered


def _source_updates(
    effective: Mapping[str, Any],
    source_document: Mapping[str, Any],
    tuning: Mapping[str, Mapping[str, float | int]],
) -> dict[str, Any]:
    """Build leaf-only updates for the source preset and its Hydra groups."""
    source_updates: dict[tuple[Any, ...], Any] = {}
    group_updates: dict[Path, dict[tuple[Any, ...], Any]] = {}

    def add_source(path: tuple[Any, ...], value: Any) -> None:
        source_updates[path] = value

    def add_group(path: Path, update_path: tuple[Any, ...], value: Any) -> None:
        group_updates.setdefault(path, {})[update_path] = value

    def source_has(path: tuple[Any, ...]) -> bool:
        current: Any = source_document
        for part in path:
            if (
                (isinstance(current, Mapping) and part in current)
                or (isinstance(current, list) and isinstance(part, int) and part < len(current))
            ):
                current = current[part]
            else:
                return False
        return True

    # Policy tuning and rollout controls are owned by the selected launch file.
    camera = effective.get("camera")
    if isinstance(camera, Mapping):
        for field in ("head_capture_id", "left_capture_id", "right_capture_id"):
            path = ("camera", field)
            if field in camera and source_has(path):
                add_source(path, camera[field])
    for field, value in tuning.get("policy", {}).items():
        path = ("policy", field)
        if source_has(path):
            add_source(path, value)
    rollout = effective.get("policy", {}).get("infer_rollout") if isinstance(effective.get("policy"), Mapping) else None
    if isinstance(rollout, Mapping):
        for field in ("enabled", "required", "output_dir"):
            path = ("policy", "infer_rollout", field)
            if field in rollout and source_has(path):
                add_source(path, rollout[field])

    defaults = source_document.get("defaults", [])
    default_groups: dict[str, Path] = {}
    if isinstance(defaults, list):
        for item in defaults:
            if not isinstance(item, Mapping):
                continue
            for raw_group, raw_name in item.items():
                group = str(raw_group).lstrip("/").split("@", maxsplit=1)[0]
                if group in {"devices", "devices/infer"} and isinstance(raw_name, str):
                    default_groups["devices"] = Path(group) / f"{raw_name}.yaml"

    effective_devices = effective.get("devices")
    if isinstance(effective_devices, Mapping):
        effective_devices = effective_devices.get("devices")
    if isinstance(effective_devices, list):
        devices_relative = default_groups.get("devices")

        def add_device_update(index: int, update_path: tuple[Any, ...], value: Any) -> None:
            path = ("devices", index, *update_path)
            if devices_relative is not None:
                add_group(devices_relative, path, value)
            elif source_has(path) or value is _DELETE_YAML_NODE:
                add_source(path, value)

        for index, device in enumerate(effective_devices):
            if not isinstance(device, Mapping):
                continue
            if str(device.get("class")) == "RizonRobot":
                for field in ("robot_sn", "network_interface_whitelist"):
                    if field in device:
                        add_device_update(index, (field,), device[field])
                gripper = device.get("gripper")
                if isinstance(gripper, Mapping) and "port" in gripper:
                    add_device_update(index, ("gripper", "port"), gripper["port"])
            elif str(device.get("class")) in {"WebCameraDevice", "WebCameraOpenPi"}:
                # Preserve the source capture_id interpolation. Physical role
                # bindings are persisted in the launch file's camera block.
                add_device_update(
                    index,
                    ("_control_panel_capture_binding",),
                    _DELETE_YAML_NODE,
                )

    return {
        "source": source_updates,
        "groups": tuple(
            (path, updates)
            for path, updates in group_updates.items()
            if updates
        ),
    }


def _patch_yaml_bytes(
    content: bytes,
    updates: Mapping[tuple[Any, ...], Any],
    path: Path,
) -> bytes:
    """Patch scalar/sequence YAML nodes while retaining comments and layout."""
    if not updates:
        return content
    try:
        text = content.decode("utf-8")
        root = yaml.compose(text)
    except (UnicodeError, yaml.YAMLError) as error:
        raise GeneratedConfigExportError(f"could not parse YAML config: {path}") from error
    if root is None:
        raise GeneratedConfigExportError(f"YAML config is empty: {path}")

    nodes: dict[tuple[Any, ...], Any] = {}
    mapping_entries: dict[tuple[Any, ...], list[tuple[Any, Any]]] = {}

    def visit(node: Any, node_path: tuple[Any, ...]) -> None:
        nodes[node_path] = node
        if isinstance(node, yaml.nodes.MappingNode):
            for key_node, value_node in node.value:
                if not isinstance(key_node, yaml.nodes.ScalarNode):
                    continue
                child_path = node_path + (key_node.value,)
                mapping_entries.setdefault(child_path, []).append((key_node, value_node))
                visit(value_node, child_path)
        elif isinstance(node, yaml.nodes.SequenceNode):
            for index, child in enumerate(node.value):
                child_path = node_path + (index,)
                visit(child, child_path)

    visit(root, ())
    replacements: list[tuple[int, int, str]] = []
    pending = dict(updates)

    def render_value(value: Any) -> str:
        try:
            rendered = yaml.safe_dump(
                value,
                allow_unicode=False,
                default_flow_style=True,
                sort_keys=False,
            ).strip()
        except yaml.YAMLError as error:
            raise GeneratedConfigExportError(f"could not serialize YAML update: {path}") from error
        # PyYAML emits a document terminator for some scalar values when they
        # are dumped as a standalone document.  It is not part of the value.
        rendered = rendered.removesuffix("\n...")
        return rendered

    for update_path, value in updates.items():
        if value is _DELETE_YAML_NODE:
            for key_node, value_node in mapping_entries.get(update_path, []):
                line_start = text.rfind("\n", 0, key_node.start_mark.index) + 1
                if text[line_start : key_node.start_mark.index].strip():
                    raise GeneratedConfigExportError(
                        f"cannot delete flow-style YAML field: {'.'.join(map(str, update_path))}"
                    )
                line_end = text.find("\n", value_node.end_mark.index)
                end = len(text) if line_end < 0 else line_end + 1
                replacements.append((line_start, end, ""))
            pending.pop(update_path, None)
            continue
        node = nodes.get(update_path)
        if node is None:
            continue
        try:
            current_value = yaml.safe_load(text[node.start_mark.index : node.end_mark.index])
        except yaml.YAMLError:
            current_value = object()
        if current_value == value:
            pending.pop(update_path, None)
            continue
        original_segment = text[node.start_mark.index : node.end_mark.index]
        if "\n" in original_segment:
            try:
                block = yaml.safe_dump(
                    value,
                    allow_unicode=False,
                    default_flow_style=False,
                    sort_keys=False,
                ).strip()
            except yaml.YAMLError as error:
                raise GeneratedConfigExportError(
                    f"could not serialize YAML update: {path}"
                ) from error
            original_lines = original_segment.splitlines()
            continuation_indent = next(
                (
                    len(line) - len(line.lstrip(" "))
                    for line in original_lines[1:]
                    if line.strip()
                ),
                0,
            )
            rendered_lines = block.splitlines()
            rendered = rendered_lines[0]
            if continuation_indent and len(rendered_lines) > 1:
                rendered += "\n" + "\n".join(
                    " " * continuation_indent + line
                    for line in rendered_lines[1:]
                )
            elif len(rendered_lines) > 1:
                rendered = "\n".join(rendered_lines)
        else:
            rendered = render_value(value)
        trailing_whitespace = original_segment[len(original_segment.rstrip(" \t\r\n")) :]
        if trailing_whitespace:
            rendered += trailing_whitespace
        replacements.append((node.start_mark.index, node.end_mark.index, rendered))
        pending.pop(update_path, None)

    # Add a missing leaf to an existing mapping.  This is used for discovered
    # camera capture paths, which are intentionally absent from many legacy
    # device group files.
    insertions: list[tuple[int, str]] = []
    for update_path, value in pending.items():
        if not update_path:
            raise GeneratedConfigExportError(f"cannot patch YAML root: {path}")
        parent_path = update_path[:-1]
        key = update_path[-1]
        parent = nodes.get(parent_path)
        if not isinstance(parent, yaml.nodes.MappingNode) or not isinstance(key, str):
            raise GeneratedConfigExportError(
                f"YAML field does not exist and cannot be inserted: {'.'.join(map(str, update_path))}"
            )
        rendered = render_value(value)
        child_indent = None
        if parent.value:
            first_key = parent.value[0][0]
            line_start = text.rfind("\n", 0, first_key.start_mark.index) + 1
            child_indent = len(text[line_start:first_key.start_mark.index])
        if child_indent is None:
            line_start = text.rfind("\n", 0, parent.start_mark.index) + 1
            parent_indent = len(text[line_start:parent.start_mark.index])
            child_indent = parent_indent + 2
        line_end = text.find("\n", parent.end_mark.index)
        if line_end < 0:
            line_end = len(text)
        insert_at = line_end + (1 if line_end < len(text) else 0)
        insertions.append((insert_at, " " * child_indent + f"{key}: {rendered}\n"))

    for start, end, rendered in sorted(replacements, reverse=True):
        text = text[:start] + rendered + text[end:]
    for index, insertion in sorted(insertions, reverse=True):
        text = text[:index] + insertion + text[index:]
    return text.encode("utf-8")


def _generated_root(config_dir: Path) -> Path:
    """Return the shared Hydra config root used by device_starter."""
    root = Path(config_dir)
    if not root.exists() or not root.is_dir() or root.is_symlink():
        raise GeneratedConfigExportError(f"config_dir must be an existing non-symlink directory: {root}")
    return root.resolve(strict=True)


def _assert_safe_target(target: Path, *, root: Path) -> None:
    if target.parent != root or target.suffix != ".yaml":
        raise GeneratedConfigExportError("generated config target escaped its root")
    if target.exists() or target.is_symlink():
        try:
            info = target.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise GeneratedConfigExportError(f"unsafe generated config target: {target}")


def _assert_safe_source_target(target: Path, *, root: Path) -> None:
    """Reject source paths that escape the config tree or traverse symlinks."""
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise GeneratedConfigExportError("source config target escaped its root") from error
    if target.suffix not in {".yaml", ".yml"} or not relative.parts:
        raise GeneratedConfigExportError("source config target must be a YAML file")
    current = root
    for component in relative.parts[:-1]:
        current /= component
        try:
            info = current.lstat()
        except FileNotFoundError as error:
            raise GeneratedConfigExportError(
                f"source config directory does not exist: {current}"
            ) from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise GeneratedConfigExportError(f"unsafe source config directory: {current}")
    if target.is_symlink():
        raise GeneratedConfigExportError(f"unsafe source config target: {target}")


def _read_existing_regular_file(path: Path) -> bytes | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise GeneratedConfigExportError(f"unsafe generated config target: {path}") from error
        raise
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise GeneratedConfigExportError(f"unsafe generated config target: {path}")
        if info.st_size > _MAX_CONFIG_BYTES:
            raise GeneratedConfigExportError(f"existing generated config is too large: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_CONFIG_BYTES:
                raise GeneratedConfigExportError(
                    f"existing generated config is too large: {path}"
                )
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _config_references_payload(target: Path, payload: bytes) -> bool:
    """Return false only when a failed publish is known not to be visible."""
    try:
        return _read_existing_regular_file(target) == payload
    except BaseException:
        # A concurrent replacement or read failure makes publication status
        # uncertain. Preserve the immutable calibration snapshot in that case.
        return True


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_atomically(
    target: Path,
    payload: bytes,
    *,
    overwrite: bool,
    safe_target_check: Callable[..., None] = _assert_safe_target,
) -> Path | None:
    """Publish YAML through a same-directory temporary file without sidecars."""
    root = target.parent
    existing = _read_existing_regular_file(target)
    if existing is not None and not overwrite:
        raise GeneratedConfigExportError(
            f"generated config already exists; use overwrite explicitly: {target.name}"
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=root,
    )
    temporary = Path(temporary_name)
    committed = False
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        if existing is None:
            try:
                os.link(temporary, target, follow_symlinks=False)
            except FileExistsError as error:
                raise GeneratedConfigExportError(
                    f"generated config already exists; use overwrite explicitly: {target.name}"
                ) from error
            committed = True
            temporary.unlink()
            _fsync_directory(root)
            return None

        safe_target_check(target, root=root)
        os.replace(temporary, target)
        committed = True
        _fsync_directory(root)
        return None
    except BaseException as error:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        if committed:
            raise _GeneratedConfigPublishedError(
                "generated config may already be published: " + target.name
            ) from error
        raise
