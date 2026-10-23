from __future__ import annotations

import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np

from incubator.control_panel import server
from incubator.control_panel.recording import validate_recording_camera_mapping
from incubator.control_panel.recording_replay import RecordingReplayService
from incubator.control_panel.runtime_config import compose_runtime_config, validate_runtime_config
from incubator.control_panel.workflow import WorkflowLifecycle


def _camera_config(*, mapping: bool = True) -> dict:
    rollout = {
        "enabled": True,
        "required": True,
        "output_dir": "data/rollouts",
    }
    if mapping:
        rollout.update(
            {
                "head_camera": "WebCameraOpenPi_2",
                "left_camera": "WebCameraOpenPi_0",
                "right_camera": "WebCameraOpenPi_4",
            }
        )
    return {
        "devices": [
            {"class": "WebCameraOpenPi", "device_class_id": 0},
            {"class": "WebCameraOpenPi", "device_class_id": 4},
            {"class": "WebCameraOpenPi", "device_class_id": 2},
        ],
        "policy": {"infer_rollout": rollout},
    }


def test_camera_mapping_is_legacy_compatible_but_partial_mapping_fails_closed() -> None:
    legacy = _camera_config(mapping=False)
    assert validate_recording_camera_mapping(legacy) == []

    partial = _camera_config(mapping=False)
    partial["policy"]["infer_rollout"]["head_camera"] = "WebCameraOpenPi_2"
    errors = validate_recording_camera_mapping(partial)
    assert any("left_camera is required" in error for error in errors)
    assert any("right_camera is required" in error for error in errors)

    duplicate = _camera_config()
    duplicate["policy"]["infer_rollout"]["right_camera"] = "WebCameraOpenPi_0"
    assert any(
        "left_camera and right_camera" in error
        for error in validate_recording_camera_mapping(duplicate)
    )


def test_recommended_config_uses_native_openpi_camera_contract() -> None:
    # The OpenPI launch file is a shared source selected explicitly by both
    # device_starter and Control Panel; it is not the panel's implicit default.
    config = compose_runtime_config('policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi')
    assert validate_runtime_config(config) == []
    assert [
        f"{device['class']}_{device['device_class_id']}"
        for device in config["devices"]
    ] == ["WebCameraOpenPi_0", "WebCameraOpenPi_1", "RizonRobot_2"]
    assert config["policy"]["obs"]["devices"] == [
        "WebCameraOpenPi_0",
        "WebCameraOpenPi_1",
        "RizonRobot_2",
    ]


def test_workflow_enqueue_propagates_camera_mapping(tmp_path: Path) -> None:
    captured = []
    workflow = WorkflowLifecycle(
        "fixture",
        _camera_config(),
        repo_root=tmp_path,
        conversion_enqueue=lambda receipt, source_root, run_config: captured.append(
            (dict(receipt), source_root, dict(run_config))
        ),
    )
    coordinator = workflow._new_recording_coordinator()
    assert coordinator._conversion_enqueue is not None
    coordinator._conversion_enqueue({"episode_id": "episode-one"})
    assert captured == [
        (
            {"episode_id": "episode-one"},
            (tmp_path / "data/rollouts").resolve(),
            {
                "config_name": "fixture",
                "head_camera": "WebCameraOpenPi_2",
                "left_camera": "WebCameraOpenPi_0",
                "right_camera": "WebCameraOpenPi_4",
            },
        )
    ]


def test_reconcile_defers_missing_mapping_to_finalized_contract(tmp_path: Path) -> None:
    calls = []

    class Jobs:
        def reconcile_source_root(self, source_root, *, run_config):
            calls.append((source_root, dict(run_config)))
            return 1

    controller = object.__new__(server.DeviceController)
    controller.selected_config = "legacy"
    controller.logs = server.LogStore()
    controller.conversion_jobs = Jobs()
    config = _camera_config(mapping=False)
    config["policy"]["infer_rollout"]["output_dir"] = str(tmp_path / "rollouts")

    assert controller._reconcile_conversion_sources(config) == 1
    assert len(calls) == 1
    assert calls[0][1]['left_camera'] is None
    assert calls[0][1]['right_camera'] is None
    assert any(
        "using the finalized data contract" in item["message"]
        for item in controller.logs.query()["items"]
    )


def test_reconcile_scans_flat_and_dated_recording_roots(tmp_path: Path) -> None:
    calls = []

    class Jobs:
        def reconcile_source_root(self, source_root, *, run_config):
            calls.append((Path(source_root), dict(run_config)))
            return 1

    root = tmp_path / "rollouts"
    root.mkdir()
    (root / "20260715").mkdir()
    controller = object.__new__(server.DeviceController)
    controller.selected_config = "fixture"
    controller.logs = server.LogStore()
    controller.conversion_jobs = Jobs()
    config = _camera_config()
    config["policy"]["infer_rollout"]["output_dir"] = str(root)

    assert controller._reconcile_conversion_sources(config) == 2
    assert [call[0] for call in calls] == [root.resolve(), (root / "20260715").resolve()]
    assert all(call[1]["config_name"] == "fixture" for call in calls)


def test_state_exposes_roles_and_shm_without_inventing_nedf_topic() -> None:
    config = _camera_config()
    devices = [
        {
            "class": "WebCameraOpenPi",
            "id": camera_id,
            "running": ready,
            "ready": ready,
            "status": "running" if ready else "stopped",
            "desired_state": "running" if ready else "stopped",
        }
        for camera_id, ready in ((0, True), (4, False), (2, True))
    ]
    lifecycle = {
        "running": True,
        "monitor_running": True,
        "total_devices": 3,
        "device_manager_running": False,
        "policy_runner_running": False,
        "devices": devices,
    }
    workflow_state = {
        "runtime": {"active": False},
        "inference": {"running": False},
        "recording": {
            "configured": True,
            "enabled": True,
            "required": True,
            "status": "stopped",
        },
    }

    controller = object.__new__(server.DeviceController)
    controller._lock = threading.RLock()
    controller.starter = SimpleNamespace(get_status=lambda: lifecycle)
    controller.workflow = SimpleNamespace(observe=lambda _devices: workflow_state)
    controller.selected_config = "fixture"
    controller._base_config = None
    controller._effective_config = config
    controller._effective_bindings = {}
    controller._binding_discovery_revision = None
    controller._gripper_state_locked = lambda _devices, _workflow: {"active": False}
    controller._checkpoint_state_locked = lambda _workflow: {}
    controller.conversion_jobs = SimpleNamespace(state=lambda: {})
    controller.revision = 1
    controller.started_at = "now"
    controller._force_stop_in_progress = False

    roles = controller.state()["recording"]["camera_roles"]
    assert roles["head"] == {
        "role": "head",
        "camera_id": "camera_2",
        "device_key": "WebCameraOpenPi_2",
        "topic": "/camera/color/compressed/jpg/camera_2",
        "shm_name": "WebCameraOpenPi_2_data",
        "status": "healthy",
        "reason": None,
    }
    assert roles["right"]["status"] == "configured"


def test_summary_uses_inspected_nedf_topics_for_camera_roles(tmp_path: Path) -> None:
    root = tmp_path / "rollouts"
    episode = root / "episode_0001"
    episode.mkdir(parents=True)
    (episode / "metadata.json").write_text(
        json.dumps(
            {
                "data_format": "NEDF3",
                "camera_info": {
                    "camera_2": "head",
                    "camera_1": "inhand_left",
                    "camera_0": "inhand_right",
                },
            }
        ),
        encoding="utf-8",
    )
    (episode / "episode_result.json").write_text(
        json.dumps({"episode_result": "success", "episode_index": 1}),
        encoding="utf-8",
    )
    inspection = {
        "schema_version": 1,
        "data_format": "NEDF3",
        "time_range_ns": {"start": 1, "end": 3},
        "topics": [
            {"name": "nmx/hal/camera/camera_2/rgbd", "count": 1},
            {"name": "nmx/hal/camera/camera_1/rgbd", "count": 1},
            {"name": "nmx/hal/camera/camera_0/rgbd", "count": 1},
        ],
        "cameras": [
            {"id": "camera_2", "source_device": "WebCameraOpenPi_2", "count": 1},
            {"id": "camera_1", "source_device": "WebCameraOpenPi_0", "count": 1},
            {"id": "camera_0", "source_device": "WebCameraOpenPi_4", "count": 1},
        ],
        "joints": {},
        "issues": [],
    }

    def runner(*_args, **_kwargs):
        command = _args[0]
        output = Path(command[command.index("--timeline-output") + 1])
        with output.open("wb") as stream:
            np.savez(stream, cache_schema_version=np.asarray([1], dtype=np.int16))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(inspection),
            stderr="",
        )

    service = RecordingReplayService(
        lambda: root,
        camera_roles_provider=lambda: {
            "head": "WebCameraOpenPi_2",
            "left": "WebCameraOpenPi_0",
            "right": "WebCameraOpenPi_4",
        },
        cache_root=tmp_path / "cache",
        process_runner=runner,
    )
    roles = service.summary("episode_0001")["camera_roles"]
    assert roles["head"]["topic"] == "nmx/hal/camera/camera_2/rgbd"
    assert roles["head"]["shm_name"] == "WebCameraOpenPi_2_data"
    assert roles["head"]["status"] == "healthy"


def test_summary_prefers_recorded_logical_camera_ids() -> None:
    service = object.__new__(RecordingReplayService)
    inspection = {
        "data_format": "NEDF3",
        "topics": [
            {"name": "nmx/hal/camera/camera_0/rgbd", "count": 5},
            {"name": "nmx/hal/camera/camera_1/rgbd", "count": 5},
            {"name": "nmx/hal/camera/camera_2/rgbd", "count": 5},
        ],
        "cameras": [
            {"id": "camera_0", "source_device": "WebCameraOpenPi_4", "count": 5},
            {"id": "camera_1", "source_device": "WebCameraOpenPi_0", "count": 5},
            {"id": "camera_2", "source_device": "WebCameraOpenPi_2", "count": 5},
        ],
    }
    roles = service._camera_roles(
        inspection,
        {
            "head": "old-head",
            "left": "old-left",
            "right": "old-right",
        },
    )

    assert roles["head"]["camera_id"] == "camera_2"
    assert roles["head"]["device_key"] == "WebCameraOpenPi_2"
    assert roles["left"]["topic"] == "nmx/hal/camera/camera_1/rgbd"
    assert roles["right"]["status"] == "healthy"


def test_summary_uses_nedf3_camera_labels_and_topic_numbering() -> None:
    service = object.__new__(RecordingReplayService)
    inspection = {
        "data_format": "NEDF3",
        "topics": [
            {"name": "nmx/hal/camera/camera_0/rgbd", "count": 5},
            {"name": "nmx/hal/camera/camera_1/rgbd", "count": 5},
            {"name": "nmx/hal/camera/camera_2/rgbd", "count": 5},
        ],
        "cameras": [
            {"id": "camera_0", "source_device": "right_wrist", "count": 5},
            {"id": "camera_1", "source_device": "left_wrist", "count": 5},
            {"id": "camera_2", "source_device": "head", "count": 5},
        ],
    }

    roles = service._camera_roles(
        inspection,
        {
            "head": "stale-head-device",
            "left": "stale-left-device",
            "right": "stale-right-device",
        },
    )

    assert roles["head"]["camera_id"] == "camera_2"
    assert roles["left"]["camera_id"] == "camera_1"
    assert roles["right"]["camera_id"] == "camera_0"
    assert roles["head"]["topic"] == "nmx/hal/camera/camera_2/rgbd"
    assert roles["left"]["topic"] == "nmx/hal/camera/camera_1/rgbd"
    assert roles["right"]["topic"] == "nmx/hal/camera/camera_0/rgbd"
    assert all(role["status"] == "healthy" for role in roles.values())



def test_create_app_wires_selected_camera_roles_into_default_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}

    def replay_factory(
        root_provider,
        *,
        camera_roles_provider,
        published_shard_provider,
        inspector_python,
        inspector_python_provider,
    ):
        captured["root_provider"] = root_provider
        captured["camera_roles_provider"] = camera_roles_provider
        captured["published_shard_provider"] = published_shard_provider
        captured["inspector_python"] = inspector_python
        captured["inspector_python_provider"] = inspector_python_provider
        return object()

    monkeypatch.setattr(server, "RecordingReplayService", replay_factory)
    app = server.create_app(config_dir=tmp_path / "configs")
    try:
        assert callable(captured["root_provider"])
        assert captured["camera_roles_provider"]() == {
            "head": None,
            "left": None,
            "right": None,
        }
        assert callable(captured["published_shard_provider"])
        assert captured["inspector_python"] is None
        assert callable(captured["inspector_python_provider"])
    finally:
        app.config["STATE_SYNC"].close()
        app.config["DEVICE_CONTROLLER"].shutdown()
