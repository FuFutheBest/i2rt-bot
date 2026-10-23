"""Tests for explicit, isolated native Hydra config exports."""

from __future__ import annotations

from pathlib import Path
import json

import pytest
import yaml

from incubator.control_panel import generated_config_export
from incubator.control_panel.generated_config_export import (
    GeneratedConfigExportError,
    export_generated_config,
)
from incubator.control_panel.runtime_config import compose_runtime_config


def _effective_config() -> dict:
    return {
        "devices": [{"class": "RizonRobot", "device_class_id": 2}],
        "camera": {"left_capture_id": 0, "right_capture_id": 4},
        "policy": {
            "class": "OpenPiUmiPolicy",
            "fps": 4.0,
            "control_fps": 30.0,
        },
        "action_executor": {"type": "ActionExecutor", "execution_fps": 30.0},
        "device_manager": {"type": "DeviceManager", "target_fps": 30.0},
    }


def test_export_writes_new_hydra_config_with_auditable_provenance(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    source = config_dir / "openpi" / "config_infer_fisheye_umi3_dual_openpi.yaml"
    source.parent.mkdir(parents=True)
    source.write_text("policy: {class: OpenPiUmiPolicy}\n", encoding="utf-8")
    effective = _effective_config()

    result = export_generated_config(
        effective,
        config_dir=config_dir,
        source_config_name="policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
        revision=17,
        generated_id="session_20260828",
        bindings={"camera": {"left_capture_id": 7}, "robot": {"serial": "RZ-1"}},
        calibration_artifacts={"left": {"status": "valid", "sha256": "a" * 64}},
        tuning_overrides={
            "policy": {"fps": 5.0, "control_fps": 40.0, "chunk_length": 32},
        },
    )

    assert result.config_name == "generated_session_20260828"
    assert result.path == config_dir / "generated_session_20260828.yaml"
    assert result.revision == "17"
    assert result.backup_path is None
    assert len(result.content_sha256) == 64
    assert source.read_text(encoding="utf-8") == "policy: {class: OpenPiUmiPolicy}\n"
    assert effective == _effective_config()

    document = yaml.safe_load(result.path.read_text(encoding="utf-8"))
    assert document["camera"] == {"left_capture_id": 0, "right_capture_id": 4}
    assert document["policy"]["fps"] == 5.0
    assert document["policy"]["control_fps"] == 40.0
    assert document["policy"]["chunk_length"] == 32
    assert document["action_executor"]["execution_fps"] == 30.0
    assert document["device_manager"]["target_fps"] == 30.0
    assert document["control_panel_export"] == {
        "schema_version": 1,
        "generated_at": result.generated_at,
        "source_config_name": "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
        "source_revision": "17",
        "bindings": {"camera": {"left_capture_id": 7}, "robot": {"serial": "RZ-1"}},
        "calibration_artifacts": {"left": {"status": "valid", "sha256": "a" * 64}},
        "tuning_overrides": {
            "policy": {"fps": 5.0, "control_fps": 40.0, "chunk_length": 32},
        },
    }
    assert result.as_dict()["path"] == str(result.path)

    recomposed = compose_runtime_config(result.config_name, config_dir=config_dir)
    assert recomposed["policy"]["fps"] == 5.0
    assert recomposed["policy"]["control_fps"] == 40.0
    assert recomposed["action_executor"]["execution_fps"] == 30.0
    assert recomposed["device_manager"]["target_fps"] == 30.0


def test_export_normalizes_removed_gripper_websocket_transport_on_direct_calls(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    effective = _effective_config()
    effective["devices"] = [
        {
            "class": "RizonRobot",
            "device_class_id": 2,
            "gripper": {
                "backend": "lktech_websocket",
                "server": {"host": "127.0.0.1", "port": 8765},
            },
        }
    ]

    result = export_generated_config(
        effective,
        config_dir=config_dir,
        source_config_name="policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
        revision=1,
        generated_id="direct_transport_migration",
    )

    generated = yaml.safe_load(result.path.read_text(encoding="utf-8"))
    gripper = generated["devices"][0]["gripper"]
    assert gripper["backend"] == "lktech"
    assert "server" not in gripper
    assert generated["control_panel_export"]["legacy_transport_migrations"] == [
        "legacy gripper backend lktech_websocket was normalized to direct lktech"
    ]
    assert effective["devices"][0]["gripper"]["backend"] == "lktech_websocket"


def test_export_normalizes_removed_gripper_service_fields(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    effective = _effective_config()
    effective["devices"] = [
        {
            "class": "RizonRobot",
            "device_class_id": 2,
            "gripper": {
                "backend": "lktech_websocket",
                "server": {"host": "127.0.0.1", "port": 8765},
                "enabled": [False, False],
            },
        }
    ]

    result = export_generated_config(
        effective,
        config_dir=config_dir,
        source_config_name="policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
        revision=1,
        generated_id="direct_only",
    )

    generated_gripper = yaml.safe_load(result.path.read_text(encoding="utf-8"))[
        "devices"
    ][0]["gripper"]
    assert generated_gripper["backend"] == "lktech"
    assert "server" not in generated_gripper
    # The caller's effective mapping is copied before migration.
    assert effective["devices"][0]["gripper"]["backend"] == "lktech_websocket"
    assert "server" in effective["devices"][0]["gripper"]


@pytest.mark.parametrize("flat_backend", ["lktech", ""])
def test_export_migrates_flat_legacy_gripper_server_without_mutating_input(
    tmp_path: Path, flat_backend: str
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    effective = _effective_config()
    effective["devices"] = [
        {
            "class": "RizonRobot",
            "device_class_id": 2,
            "gripper_backend": flat_backend,
            "utilize_gripper": False,
            "gripper_server": {"host": "127.0.0.1", "port": 8765},
        }
    ]

    result = export_generated_config(
        effective,
        config_dir=config_dir,
        source_config_name="policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
        revision=1,
        generated_id=f"flat_transport_{'default' if not flat_backend else 'direct'}",
    )

    generated = yaml.safe_load(result.path.read_text(encoding="utf-8"))
    generated_device = generated["devices"][0]
    assert generated_device["gripper_backend"] == "lktech"
    assert "gripper_server" not in generated_device
    assert generated["control_panel_export"]["legacy_transport_migrations"] == [
        "legacy gripper server fields were ignored (default direct lktech)"
    ]
    assert effective["devices"][0]["gripper_backend"] == flat_backend
    assert "gripper_server" in effective["devices"][0]


def test_export_preserves_gripper_calibration_paths_without_copying(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    calibration_dir = tmp_path / "calibration" / "lktech_gripper"
    calibration_dir.mkdir(parents=True)
    left_source = calibration_dir / "left.json"
    right_source = calibration_dir / "right.json"
    left_calibration = {
        "motor_id": 1,
        "port": "/dev/ttyUSB0",
        "min_encoder": 100,
        "max_encoder": 200,
    }
    right_calibration = {
        "motor_id": 1,
        "port": "/dev/ttyUSB1",
        "min_encoder": 300,
        "max_encoder": 400,
    }
    left_source.write_text(json.dumps(left_calibration), encoding="utf-8")
    right_source.write_text(json.dumps(right_calibration), encoding="utf-8")
    effective = _effective_config()
    effective["devices"] = [
        {
            "class": "RizonRobot",
            "device_class_id": 2,
            "gripper": {
                "backend": "lktech",
                "enabled": [True, True],
                "port": ["/dev/ttyUSB0", "/dev/ttyUSB1"],
                "motor_id": [1, 1],
                "calibration_files": [
                    "calibration/lktech_gripper/left.json",
                    "calibration/lktech_gripper/right.json",
                ],
            },
        }
    ]

    result = export_generated_config(
        effective,
        config_dir=config_dir,
        source_config_name="policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
        revision=2,
        generated_id="frozen_calibration",
    )

    generated = yaml.safe_load(result.path.read_text(encoding="utf-8"))
    files = generated["devices"][0]["gripper"]["calibration_files"]
    assert files == [
        "calibration/lktech_gripper/left.json",
        "calibration/lktech_gripper/right.json",
    ]
    assert "calibration_snapshots" not in generated["control_panel_export"]
    assert not (tmp_path / "calibration" / "control_panel").exists()

    left_source.write_text(json.dumps({"motor_id": 99}), encoding="utf-8")
    assert json.loads(left_source.read_text(encoding="utf-8")) == {"motor_id": 99}


def _gripper_effective_config() -> dict:
    effective = _effective_config()
    effective["devices"] = [
        {
            "class": "RizonRobot",
            "device_class_id": 2,
            "gripper": {
                "backend": "lktech",
                "enabled": [True, True],
                "port": ["/dev/ttyUSB0", "/dev/ttyUSB1"],
                "motor_id": [1, 1],
                "calibration_files": [
                    "calibration/lktech_gripper/left.json",
                    "calibration/lktech_gripper/right.json",
                ],
            },
        }
    ]
    return effective


def _write_calibration_pair(root: Path) -> None:
    calibration_dir = root / "calibration" / "lktech_gripper"
    calibration_dir.mkdir(parents=True)
    for side, port, minimum, maximum in (
        ("left", "/dev/ttyUSB0", 100, 200),
        ("right", "/dev/ttyUSB1", 300, 400),
    ):
        (calibration_dir / f"{side}.json").write_text(
            json.dumps(
                {
                    "motor_id": 1,
                    "port": port,
                    "min_encoder": minimum,
                    "max_encoder": maximum,
                }
            ),
            encoding="utf-8",
        )


def test_export_reports_when_config_directory_sync_fails_after_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    _write_calibration_pair(tmp_path)
    original_fsync_directory = generated_config_export._fsync_directory
    output_root = config_dir

    def fail_only_after_config_commit(directory: Path) -> None:
        if Path(directory) == output_root:
            raise OSError("simulated generated-config directory sync failure")
        original_fsync_directory(directory)

    monkeypatch.setattr(
        generated_config_export,
        "_fsync_directory",
        fail_only_after_config_commit,
    )

    with pytest.raises(GeneratedConfigExportError, match="may already be published"):
        export_generated_config(
            _gripper_effective_config(),
            config_dir=config_dir,
            source_config_name="policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
            revision=1,
            generated_id="sync_failure",
        )

    generated_path = output_root / "generated_sync_failure.yaml"
    assert generated_path.is_file()
    generated = yaml.safe_load(generated_path.read_text(encoding="utf-8"))
    assert generated["devices"][0]["gripper"]["calibration_files"] == [
        "calibration/lktech_gripper/left.json",
        "calibration/lktech_gripper/right.json",
    ]
    assert not (tmp_path / "calibration" / "control_panel").exists()


@pytest.mark.parametrize(
    "generated_id", ["../escape", "nested/name", "", "name.yaml", "*bad"]
)
def test_export_rejects_path_like_or_unsafe_generated_names(
    tmp_path: Path, generated_id: str
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    with pytest.raises(GeneratedConfigExportError, match="generated_id"):
        export_generated_config(
            _effective_config(),
            config_dir=config_dir,
            source_config_name="policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
            revision=1,
            generated_id=generated_id,
        )

    assert not (tmp_path / "escape.yaml").exists()


@pytest.mark.parametrize(
    ("effective_config", "expected"),
    [
        ({"bad": object()}, "unsupported value type"),
        ({"bad": float("nan")}, "non-finite float"),
        ({1: "bad key"}, "mapping keys"),
    ],
)
def test_export_rejects_values_that_cannot_be_safely_serialized(
    tmp_path: Path, effective_config: dict, expected: str
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    with pytest.raises(GeneratedConfigExportError, match=expected):
        export_generated_config(
            effective_config,
            config_dir=config_dir,
            source_config_name="policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
            revision=1,
            generated_id="safe",
        )

    assert not (config_dir / "generated_safe.yaml").exists()


@pytest.mark.parametrize("provenance", [{"bad": object()}, {"bad": float("inf")}])
def test_export_rejects_unsafe_binding_or_calibration_provenance(
    tmp_path: Path, provenance: dict
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    with pytest.raises(GeneratedConfigExportError):
        export_generated_config(
            _effective_config(),
            config_dir=config_dir,
            source_config_name="policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
            revision=1,
            generated_id="safe",
            bindings=provenance,
        )
    with pytest.raises(GeneratedConfigExportError):
        export_generated_config(
            _effective_config(),
            config_dir=config_dir,
            source_config_name="policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
            revision=1,
            generated_id="safe",
            calibration_artifacts=provenance,
        )


@pytest.mark.parametrize(
    ("tuning_overrides", "expected"),
    [
        ({"policy": {"params": {"ckpt_dir": "/not-allowed"}}}, "not allowed"),
        ({"policy": {"fps": 0}}, "between"),
        ({"policy": {"chunk_length": 1.5}}, "integer"),
        ({"action_executor": {"execution_fps": 40.0}}, "not allowed"),
        ({"device_manager": {"target_fps": 60.0}}, "not allowed"),
        ({"runtime": {"enable_visualizers": True}}, "not allowed"),
    ],
)
def test_export_only_applies_allowlisted_restart_aware_tuning(
    tmp_path: Path, tuning_overrides: dict, expected: str
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    with pytest.raises(GeneratedConfigExportError, match=expected):
        export_generated_config(
            _effective_config(),
            config_dir=config_dir,
            source_config_name="policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
            revision=1,
            generated_id="safe",
            tuning_overrides=tuning_overrides,
        )


def test_export_rejects_source_path_traversal_and_symlinked_target(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    target = config_dir / "generated_safe.yaml"
    outside = tmp_path / "outside.yaml"
    outside.write_text("outside\n", encoding="utf-8")
    target.symlink_to(outside)

    for source_config_name in (
        "../openpi",
        ".",
        "..",
        "devices/demo",
        "legacy/foo/bar",
    ):
        with pytest.raises(GeneratedConfigExportError, match="source_config_name"):
            export_generated_config(
                _effective_config(),
                config_dir=config_dir,
                source_config_name=source_config_name,
                revision=1,
                generated_id="safe",
            )

    with pytest.raises(
        GeneratedConfigExportError, match="unsafe generated config target"
    ):
        export_generated_config(
            _effective_config(),
            config_dir=config_dir,
            source_config_name="policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
            revision=1,
            generated_id="safe",
            overwrite=True,
        )

    assert outside.read_text(encoding="utf-8") == "outside\n"


def test_export_accepts_native_openpi_source_name(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    result = export_generated_config(
        _effective_config(),
        config_dir=config_dir,
        source_config_name="policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
        revision=1,
        generated_id="openpi-source",
    )

    assert result.source_config_name == "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi"


def test_export_overwrite_publishes_atomically_without_backup(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    export_generated_config(
        _effective_config(),
        config_dir=config_dir,
        source_config_name="policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
        revision="binding-1",
        generated_id="operator",
    )
    effective = _effective_config()
    effective["camera"] = {"left_capture_id": 9}
    second = export_generated_config(
        effective,
        config_dir=config_dir,
        source_config_name="policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
        revision="binding-2",
        generated_id="operator",
        overwrite=True,
    )

    assert second.backup_path is None
    assert yaml.safe_load(second.path.read_text(encoding="utf-8"))["camera"] == {
        "left_capture_id": 9
    }
    assert not list(second.path.parent.glob("generated_operator.yaml.*.bak"))
    assert not list(second.path.parent.glob(".generated_operator.yaml.*.tmp"))


def test_existing_config_growth_after_initial_size_check_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "generated.yaml"
    existing.write_bytes(b"small")
    existing_inode = existing.stat().st_ino
    original_fstat = generated_config_export.os.fstat
    expanded = False

    def expand_after_size_check(descriptor: int):
        nonlocal expanded
        info = original_fstat(descriptor)
        if not expanded and info.st_ino == existing_inode:
            with existing.open("ab") as stream:
                stream.write(b"x" * generated_config_export._MAX_CONFIG_BYTES)
            expanded = True
        return info

    monkeypatch.setattr(generated_config_export.os, "fstat", expand_after_size_check)

    with pytest.raises(GeneratedConfigExportError, match="too large"):
        generated_config_export._read_existing_regular_file(existing)

    assert expanded is True
