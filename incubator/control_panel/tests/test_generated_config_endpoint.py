"""End-to-end API contracts for Control Panel generated config exports."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from incubator.control_panel import gripper_control, server
from incubator.control_panel.hardware_discovery import HardwareSnapshot
from incubator.control_panel.runtime_config import compose_runtime_config


class _StaticDiscovery:
    def discover(self) -> HardwareSnapshot:
        return HardwareSnapshot(
            revision="empty-hardware-revision",
            generated_at="2026-08-28T00:00:00+00:00",
            interfaces=[],
            cameras=[],
            robots=[],
            providers=[],
            warnings=[],
        )


class _CameraDiscovery:
    def discover(self) -> HardwareSnapshot:
        return HardwareSnapshot(
            revision="camera-hardware-revision",
            generated_at="2026-09-02T00:00:00+00:00",
            interfaces=[],
            cameras=[
                {
                    "id": "camera-video1",
                    "devnode": "/dev/video1",
                    "stable_path": "/dev/v4l/by-path/camera-one-video-index0",
                    "name": "Camera One",
                    "capture_capable": True,
                    "source": "test",
                },
                {
                    "id": "camera-video3",
                    "devnode": "/dev/video3",
                    "stable_path": "/dev/v4l/by-path/camera-three-video-index0",
                    "name": "Camera Three",
                    "capture_capable": True,
                    "source": "test",
                },
            ],
            robots=[],
            providers=[],
            warnings=[],
        )


def _write_exportable_config(config_dir: Path) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    source = config_dir / "openpi.yaml"
    source.write_text(
        """# @package _global_
devices:
  - class: BaseDevice
    device_class_id: 0
    data_shape: [2]
    data_dtype: float32
    fps: 10
policy:
  class: OpenPiPolicy
  fps: 4.0
  control_fps: 30.0
  chunk_length: 48
action_executor:
  type: ActionExecutor
  execution_fps: 30.0
device_manager:
  type: DeviceManager
  target_fps: 30.0
""",
        encoding="utf-8",
    )
    return source


def _write_camera_role_config(config_dir: Path) -> tuple[Path, Path]:
    source = config_dir / "policy_infer" / "openpi" / "cameras.yaml"
    devices = config_dir / "devices" / "infer" / "camera_pair.yaml"
    source.parent.mkdir(parents=True, exist_ok=True)
    devices.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """# @package _global_
defaults:
  - /devices/infer@devices: camera_pair
  - _self_
# Keep this camera comment.
camera:
  left_capture_id: 1
  right_capture_id: 3
policy:
  class: OpenPiPolicy
  fps: 4.0
  control_fps: 30.0
  chunk_length: 48
  obs:
    devices:
      - WebCameraOpenPi_0
      - WebCameraOpenPi_1
action_executor:
  type: ActionExecutor
  execution_fps: 30.0
device_manager:
  type: DeviceManager
""",
        encoding="utf-8",
    )
    devices.write_text(
        """devices:
  - class: WebCameraOpenPi
    device_class_id: 0
    capture_id: ${camera.left_capture_id}
    _control_panel_capture_binding: stale
    data_shape: [360, 640, 3]
    data_dtype: uint8
    fps: 30
  - class: WebCameraOpenPi
    device_class_id: 1
    capture_id: ${camera.right_capture_id}
    data_shape: [360, 640, 3]
    data_dtype: uint8
    fps: 30
""",
        encoding="utf-8",
    )
    return source, devices


def _write_rizon_exportable_config(config_dir: Path) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    source = config_dir / "openpi.yaml"
    source.write_text(
        """# @package _global_
devices:
  - class: RizonRobot
    device_class_id: 0
    data_shape: [44]
    data_dtype: float64
    fps: 120
    gripper:
      backend: lktech
      enabled: [true, true]
      port: [/dev/ttyUSB1, /dev/ttyUSB0]
      motor_id: [1, 1]
      calibration_files:
        - calibration/lktech_gripper/left.json
        - calibration/lktech_gripper/right.json
policy:
  class: OpenPiPolicy
  fps: 4.0
  control_fps: 30.0
  chunk_length: 48
action_executor:
  type: ActionExecutor
  execution_fps: 30.0
device_manager:
  type: DeviceManager
  target_fps: 30.0
""",
        encoding="utf-8",
    )
    return source


def _export_payload(state: dict, *, tuning_overrides: dict) -> dict:
    return {
        "generated_id": "operator_trial",
        "expected_selected_config": state["selected_config"],
        "expected_revision": state["revision"],
        "tuning_overrides": tuning_overrides,
        "overwrite": False,
    }


def test_generated_config_api_exports_stopped_effective_config_and_is_selectable(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    source = _write_exportable_config(config_dir)
    original_source = source.read_text(encoding="utf-8")
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=_StaticDiscovery(),
    )
    app.testing = True
    client = app.test_client()
    try:
        selected = client.post("/api/config/select", json={"name": "openpi"})
        assert selected.status_code == 200
        state = client.get("/api/state").get_json()
        tuning = client.get("/api/runtime/tuning").get_json()
        assert tuning["config_name"] == "openpi"
        assert tuning["selected_config"] == "openpi"
        assert tuning["revision"] == state["revision"]

        response = client.post(
            "/api/configs/generated",
            json=_export_payload(
                state,
                tuning_overrides={
                    "policy": {"fps": 5.0, "control_fps": 40.0, "chunk_length": 32},
                },
            ),
        )

        assert response.status_code == 201
        payload = response.get_json()
        assert payload["ok"] is True
        assert payload["config_name"] == "generated_operator_trial"
        assert payload["source_config_name"] == "openpi"
        assert source.read_text(encoding="utf-8") == original_source

        generated_path = config_dir / "generated_operator_trial.yaml"
        generated = yaml.safe_load(generated_path.read_text(encoding="utf-8"))
        assert generated["policy"]["fps"] == 5.0
        assert generated["policy"]["control_fps"] == 40.0
        assert generated["policy"]["chunk_length"] == 32
        assert generated["control_panel_export"]["source_revision"] == str(
            state["revision"]
        )

        recomposed = compose_runtime_config(
            "generated_operator_trial",
            config_dir=config_dir,
        )
        assert recomposed["action_executor"]["execution_fps"] == 30.0
        assert recomposed["device_manager"]["target_fps"] == 30.0

        configs = client.get("/api/configs").get_json()
        assert "generated_operator_trial" in configs["configs"]
        select_generated = client.post(
            "/api/config/select",
            json={"name": "generated_operator_trial"},
        )
        assert select_generated.status_code == 200
    finally:
        app.config["DEVICE_CONTROLLER"].shutdown()


def test_generated_config_api_rejects_stale_or_unallowlisted_mutations(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    _write_exportable_config(config_dir)
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=_StaticDiscovery(),
    )
    app.testing = True
    client = app.test_client()
    try:
        assert client.post("/api/config/select", json={"name": "openpi"}).status_code == 200
        state = client.get("/api/state").get_json()

        stale = _export_payload(state, tuning_overrides={})
        stale["expected_revision"] -= 1
        stale_response = client.post("/api/configs/generated", json=stale)
        assert stale_response.status_code == 409
        assert not (config_dir / "generated_operator_trial.yaml").exists()

        invalid = _export_payload(
            state,
            tuning_overrides={"device_manager": {"target_fps": 60.0}},
        )
        invalid_response = client.post("/api/configs/generated", json=invalid)
        assert invalid_response.status_code == 422
        assert not (config_dir / "generated_operator_trial.yaml").exists()

        explicit_execution_fps = _export_payload(
            state,
            tuning_overrides={"action_executor": {"execution_fps": 40.0}},
        )
        explicit_response = client.post(
            "/api/configs/generated", json=explicit_execution_fps
        )
        assert explicit_response.status_code == 422
        assert not (config_dir / "generated_operator_trial.yaml").exists()
    finally:
        app.config["DEVICE_CONTROLLER"].shutdown()


def test_camera_bindings_only_change_capture_ids(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    source, devices_source = _write_camera_role_config(config_dir)
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=_CameraDiscovery(),
    )
    app.testing = True
    client = app.test_client()
    try:
        selected = client.post(
            "/api/config/select",
            json={
                "name": "policy_infer/openpi/cameras",
                "discovery_revision": "camera-hardware-revision",
                "overrides": {
                    "WebCameraOpenPi:0": {"capture_device": "camera-video3"},
                    "WebCameraOpenPi:1": {"capture_device": "camera-video1"},
                },
            },
        )
        assert selected.status_code == 200
        state = selected.get_json()

        generated_response = client.post(
            "/api/configs/generated",
            json=_export_payload(state, tuning_overrides={}),
        )
        assert generated_response.status_code == 201
        generated_path = config_dir / "generated_operator_trial.yaml"
        assert "left_capture_id" in generated_path.read_text(encoding="utf-8")
        generated = compose_runtime_config(
            "generated_operator_trial",
            config_dir=config_dir,
        )
        assert generated["camera"] == {
            "left_capture_id": "/dev/v4l/by-path/camera-three-video-index0",
            "right_capture_id": "/dev/v4l/by-path/camera-one-video-index0",
        }
        assert [device["device_class_id"] for device in generated["devices"]] == [0, 1]
        assert generated["policy"]["obs"]["devices"] == [
            "WebCameraOpenPi_0",
            "WebCameraOpenPi_1",
        ]

        save_response = client.post(
            "/api/config/save",
            json={
                "expected_selected_config": "policy_infer/openpi/cameras",
                "expected_revision": state["revision"],
                "tuning_overrides": {},
            },
        )
        assert save_response.status_code == 200
        source_text = source.read_text(encoding="utf-8")
        devices_text = devices_source.read_text(encoding="utf-8")
        assert "# Keep this camera comment." in source_text
        assert yaml.safe_load(source_text)["camera"] == {
            "left_capture_id": "/dev/v4l/by-path/camera-three-video-index0",
            "right_capture_id": "/dev/v4l/by-path/camera-one-video-index0",
        }
        assert "capture_id: ${camera.left_capture_id}" in devices_text
        assert "capture_id: ${camera.right_capture_id}" in devices_text
        assert "_control_panel_capture_binding" not in devices_text
        assert "device_class_id: 0" in devices_text
        assert "device_class_id: 1" in devices_text
        assert not list(config_dir.rglob("*.bak"))

        controller = app.config["DEVICE_CONTROLLER"]
        assert [
            device["config"]["device_class_id"] for device in controller.starter.devices
        ] == [0, 1]
        assert set(controller.state()["effective_bindings"]) == {
            "WebCameraOpenPi:0",
            "WebCameraOpenPi:1",
        }
        assert controller.state()["effective_bindings"]["WebCameraOpenPi:0"][
            "capture_device"
        ]["devnode"] == "/dev/video3"
    finally:
        app.config["DEVICE_CONTROLLER"].shutdown()


def test_source_config_api_saves_effective_state_without_backup(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    source = _write_exportable_config(config_dir)
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=_StaticDiscovery(),
    )
    app.testing = True
    client = app.test_client()
    try:
        assert client.post("/api/config/select", json={"name": "openpi"}).status_code == 200
        state = client.get("/api/state").get_json()
        response = client.post(
            "/api/config/save",
            json={
                "expected_selected_config": "openpi",
                "expected_revision": state["revision"],
                "tuning_overrides": {
                    "policy": {"fps": 5.0, "control_fps": 40.0, "chunk_length": 32},
                },
            },
        )
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True
        assert payload["selected_config"] == "openpi"
        assert payload["reload_deferred"] is False
        assert payload["backup_path"] is None
        assert not list(config_dir.rglob("*.bak"))
        assert yaml.safe_load(source.read_text(encoding="utf-8"))["policy"]["fps"] == 5.0
        assert yaml.safe_load(source.read_text(encoding="utf-8"))["policy"]["control_fps"] == 40.0
        assert yaml.safe_load(source.read_text(encoding="utf-8"))["policy"]["chunk_length"] == 32
        assert yaml.safe_load(source.read_text(encoding="utf-8"))["action_executor"]["execution_fps"] == 30.0
        assert yaml.safe_load(source.read_text(encoding="utf-8"))["device_manager"]["target_fps"] == 30.0

        saved_tuning = client.get("/api/runtime/tuning").get_json()
        saved_controls = {item["key"]: item for item in saved_tuning["controls"]}
        assert saved_tuning["revision"] == payload["revision"]
        assert saved_controls["policy.fps"]["value"] == 5.0
        assert saved_controls["policy.control_fps"]["value"] == 40.0
        assert saved_controls["policy.chunk_length"]["value"] == 32
        assert "action_executor.execution_fps" not in saved_controls
    finally:
        app.config["DEVICE_CONTROLLER"].shutdown()


def test_source_config_api_saves_beside_active_device_without_reloading(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    source = _write_exportable_config(config_dir)
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=_StaticDiscovery(),
    )
    app.testing = True
    client = app.test_client()
    controller = app.config["DEVICE_CONTROLLER"]
    try:
        assert client.post("/api/config/select", json={"name": "openpi"}).status_code == 200
        state = client.get("/api/state").get_json()
        starter = controller.starter
        assert starter is not None
        original_has_active_devices = starter.has_active_devices
        starter.has_active_devices = lambda: True
        try:
            response = client.post(
                "/api/config/save",
                json={
                    "expected_selected_config": "openpi",
                    "expected_revision": state["revision"],
                    "tuning_overrides": {"policy": {"fps": 5.0}},
                },
            )
            assert response.status_code == 200
            payload = response.get_json()
            assert payload["reload_deferred"] is True
            assert payload["revision"] == state["revision"] + 1
            assert controller.starter is starter
            assert controller._effective_config["policy"]["fps"] == 4.0

            generated = client.post(
                "/api/configs/generated",
                json={
                    "generated_id": "active_rejected",
                    "expected_selected_config": "openpi",
                    "expected_revision": payload["revision"],
                    "tuning_overrides": {},
                    "overwrite": False,
                },
            )
            assert generated.status_code == 409
        finally:
            starter.has_active_devices = original_has_active_devices

        assert yaml.safe_load(source.read_text(encoding="utf-8"))["policy"]["fps"] == 5.0
        assert not (config_dir / "generated_active_rejected.yaml").exists()
    finally:
        controller.shutdown()


def test_generated_rizon_config_keeps_source_calibration_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime_root = tmp_path / "runtime-repository"
    calibration_dir = runtime_root / "calibration" / "lktech_gripper"
    calibration_dir.mkdir(parents=True)
    for side, port, low, high in (
        ("left", "/dev/ttyUSB1", 100, 200),
        ("right", "/dev/ttyUSB0", 300, 400),
    ):
        (calibration_dir / f"{side}.json").write_text(
            json.dumps(
                {
                    "motor_id": 1,
                    "port": port,
                    "min_encoder": low,
                    "max_encoder": high,
                }
            ),
            encoding="utf-8",
        )

    # The config directory intentionally lives outside the repository root.
    # Runtime still resolves the unchanged relative paths from its repository.
    config_dir = tmp_path / "external-configs"
    _write_rizon_exportable_config(config_dir)
    monkeypatch.setattr(gripper_control, "REPO_ROOT", runtime_root)
    monkeypatch.setattr(server, "rizon_shared_memory_names", lambda: [])
    app = server.create_app(config_dir=config_dir, hardware_discovery=_StaticDiscovery())
    app.testing = True
    client = app.test_client()
    try:
        selected = client.post("/api/config/select", json={"name": "openpi"})
        assert selected.status_code == 200
        state = selected.get_json()

        exported = client.post(
            "/api/configs/generated",
            json=_export_payload(state, tuning_overrides={}),
        )
        assert exported.status_code == 201
        generated_path = config_dir / "generated_operator_trial.yaml"
        generated = yaml.safe_load(generated_path.read_text(encoding="utf-8"))
        calibration_files = generated["devices"][0]["gripper"]["calibration_files"]
        assert calibration_files == [
            "calibration/lktech_gripper/left.json",
            "calibration/lktech_gripper/right.json",
        ]
        assert all((runtime_root / path).is_file() for path in calibration_files)
        assert all(not (config_dir.parent / path).exists() for path in calibration_files)
        assert not (runtime_root / "calibration" / "control_panel").exists()

        reselected = client.post(
            "/api/config/select",
            json={"name": "generated_operator_trial"},
        )
        assert reselected.status_code == 200
        gripper_state = app.config["DEVICE_CONTROLLER"].grippers.state()
        assert gripper_state["sides"]["left"]["configured"] is True
        assert gripper_state["sides"]["right"]["configured"] is True
    finally:
        app.config["DEVICE_CONTROLLER"].shutdown()
