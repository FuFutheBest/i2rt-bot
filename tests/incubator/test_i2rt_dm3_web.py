"""Hardware-free contracts for the I2RT DM3 browser interface."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from incubator.nedf3.I2RT_DM3.controller import task_output_dir
from incubator.nedf3.I2RT_DM3.process_manager import ManagedProcess
from incubator.nedf3.I2RT_DM3.web.__main__ import _is_loopback
from incubator.nedf3.I2RT_DM3.web.app import I2RTWebService, create_app
from incubator.nedf3.windows_DM3.dm3_api import DM3TaskExecution


class FakeSession:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {
            "logged_in": False,
            "login_phone": "",
            "user_id": "",
            "user_name": "",
            "task_selected": False,
            "task_execution_id": None,
            "task_name": "",
            "task_status": "",
            "task_description": "",
        }
        self.tasks = [
            DM3TaskExecution(
                id=42,
                name="测试任务",
                status="ready",
                raw={"id": 42, "name": "测试任务"},
            )
        ]

    def public_state(self) -> dict[str, Any]:
        return dict(self.state)

    def login(self, phone: str, password: str, base_url: str) -> dict[str, Any]:
        assert password
        assert base_url.startswith("https://")
        self.state.update(
            logged_in=True,
            login_phone=phone,
            user_id="user-1",
            user_name="operator",
        )
        return self.public_state()

    def logout(self) -> None:
        self.__init__()

    def fetch_tasks(self, **_kwargs: Any) -> list[DM3TaskExecution]:
        return self.tasks

    def select_task_by_index(self, index: int) -> dict[str, Any]:
        task = self.tasks[index]
        self.state.update(
            task_selected=True,
            task_execution_id=task.id,
            task_name=task.name,
            task_status=task.status,
            task_description="collect",
        )
        return self.public_state()


class FakeProcesses:
    def __init__(self) -> None:
        self.device = ManagedProcess("device")
        self.saver = ManagedProcess("saver")
        self.calibration = ManagedProcess("calibration")
        self.visualizer = ManagedProcess("visualizer")
        self.commands: list[str] = []

    def runtime_status(self) -> dict[str, Any]:
        return {
            "device": self.device.status_dict(),
            "saver": self.saver.status_dict(),
            "calibration": self.calibration.status_dict(),
            "visualizer": self.visualizer.status_dict(),
        }

    def stop_device(self) -> None:
        self.device.running = False
        self.device.ready_for_episode = False

    def stop_saver(self) -> None:
        self.saver.running = False
        self.saver.recording = False

    def start_calibration(self) -> None:
        self.calibration.running = True

    def send_saver_command(self, command: str) -> None:
        self.commands.append(command)


class FakeUploadManager:
    def __init__(self) -> None:
        self.log_callback = None
        self.auto_enabled = False
        self.configure_calls: list[dict[str, Any]] = []
        self.shutdown_called = False

    def start_auto(self) -> None:
        self.auto_enabled = True

    def stop_auto(self) -> None:
        self.auto_enabled = False

    def shutdown(self) -> None:
        self.shutdown_called = True

    def configure(self, **kwargs: Any) -> None:
        self.configure_calls.append(kwargs)

    def upload_progress_snapshot(self) -> dict[str, Any]:
        return {
            "auto_enabled": self.auto_enabled,
            "worker_running": False,
            "current_episode": None,
            "queued_count": 0,
            "performance": {"running": True, "alerts": []},
        }

    def snapshot(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            **self.upload_progress_snapshot(),
            "data_dir": "/tmp/data",
            "resource_type": "i2rt",
            "policy": {
                "protect_minutes": 10,
                "keep_latest_count": 5,
                "max_retries": 5,
                "upload_workers": 2,
                "multipart_threads": 4,
            },
            "stats": {"total": 0, "complete": 0, "queued": 0, "success": 0, "failed": 0},
            "episodes": [],
        }

    def scan(self) -> list[Any]:
        return []

    def enqueue_eligible(self, *, force: bool) -> int:
        return 2 if force else 1

    def enqueue_failed(self) -> int:
        return 3

    def refresh_after_episode(self, _path: Path) -> Any:
        return type("Result", (), {"episodes": []})()

    def enqueue_eligible_episodes(self, _episodes: list[Any], *, force: bool) -> int:
        return 0


class FakeController:
    def __init__(self) -> None:
        self.session = FakeSession()
        self.processes = FakeProcesses()
        self.device_config_name = "i2rt_data_collect_async"
        self.saver_config_path = "configs/i2rt_nedf3_saver.yaml"
        self.gripper_calibration_supported = True
        self.output_dir = "/tmp/data/i2rt"
        self.task_name = "i2rt"
        self.resource_type = "i2rt"
        self.upload_manager: FakeUploadManager | None = None
        self.shutdown_called = False
        self.visualized_episodes: list[str] = []
        self.available_visualizer_episodes = [
            {
                "path": "task/episode_0004",
                "task": "task",
                "name": "episode_0004",
                "modified_at_ms": 1_000,
            }
        ]

    def apply_selected_task_defaults(self) -> None:
        self.task_name = str(self.session.state["task_name"])
        self.output_dir = str(task_output_dir(self.task_name))

    def create_upload_manager(self, _data_dir: Path, log_callback=None) -> FakeUploadManager:
        self.upload_manager = FakeUploadManager()
        self.upload_manager.log_callback = log_callback
        self.upload_manager.configure(resource_type="i2rt")
        return self.upload_manager

    def shutdown_upload_manager(self) -> None:
        if self.upload_manager is not None:
            self.upload_manager.shutdown()
        self.upload_manager = None

    def start_device(self) -> None:
        self.processes.device.running = True
        self.processes.device.ready_for_episode = True
        self.processes.device.teleoperation_phase = "safe_hold"

    def start_gripper_calibration(self) -> None:
        self.processes.start_calibration()

    def start_saver(self) -> None:
        self.processes.saver.running = True
        self.processes.saver.ready_for_episode = True

    def visualizer_episodes(self) -> list[dict[str, Any]]:
        return self.available_visualizer_episodes

    def start_visualizer(self, relative_path: str) -> Path:
        episode = Path("/tmp/data") / relative_path
        self.visualized_episodes.append(episode.name)
        return episode

    def shutdown(self) -> None:
        self.shutdown_called = True
        self.processes.stop_saver()
        self.processes.stop_device()
        self.shutdown_upload_manager()


@pytest.fixture
def web_app():
    controller = FakeController()
    service = I2RTWebService(controller)  # type: ignore[arg-type]
    app = create_app(service=service)
    app.config.update(TESTING=True)
    yield app, controller
    service.shutdown()


def _csrf(app) -> dict[str, str]:
    return {"X-I2RT-CSRF": app.config["I2RT_CSRF_TOKEN"]}


def test_yam_gripper_mode_disables_lktech_calibration(web_app) -> None:
    app, controller = web_app
    controller.gripper_calibration_supported = False
    controller.session.state.update(logged_in=True, task_selected=True)

    state = app.test_client().get("/api/state").get_json()["data"]
    assert state["controls"]["calibration_supported"] is False
    assert state["controls"]["can_calibrate_grippers"] is False


def test_web_page_has_only_login_collection_and_upload_views(web_app) -> None:
    app, _controller = web_app
    source = app.test_client().get("/").get_data(as_text=True)

    assert "DM3 登录" in source
    assert "数据采集" in source
    assert "上传与性能监控" in source
    assert "tracker" not in source.casefold()
    assert "resource_type" in source
    assert 'id="teleoperation-badge"' in source
    assert 'id="teleoperation-fault-reason"' in source
    assert 'id="calibrate-grippers-button"' in source
    assert 'id="operation-status"' in source
    assert 'id="operation-status-label"' in source
    assert 'id="visualize-episode-button"' in source
    assert 'id="episode-picker-dialog"' in source
    assert 'id="camera-right-image"' in source
    assert 'id="camera-left-image"' in source
    assert 'id="camera-head-image"' in source
    assert 'data-task-required' in source


def test_camera_snapshot_route_returns_jpeg_and_rejects_unknown_role(
    web_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, controller = web_app
    service = app.config["I2RT_WEB_SERVICE"]
    controller.session.state["logged_in"] = True
    monkeypatch.setattr(service, "camera_snapshot", lambda role: b"jpeg-" + role.encode())

    response = app.test_client().get("/api/cameras/right/snapshot.jpg")

    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.data == b"jpeg-right"


def test_camera_stream_route_uses_mjpeg_content_type(
    web_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, controller = web_app
    service = app.config["I2RT_WEB_SERVICE"]
    controller.session.state["logged_in"] = True
    monkeypatch.setattr(
        service,
        "camera_stream",
        lambda role: iter([b"--frame\r\n" + role.encode()]),
    )

    response = app.test_client().get("/api/cameras/left/stream.mjpg")

    assert response.status_code == 200
    assert response.mimetype == "multipart/x-mixed-replace"
    assert response.data == b"--frame\r\nleft"


def test_camera_snapshot_validates_role_before_reading_hardware(web_app) -> None:
    app, controller = web_app
    controller.session.state["logged_in"] = True

    response = app.test_client().get("/api/cameras/unknown/snapshot.jpg")

    assert response.status_code == 404


def test_camera_snapshot_requires_login_and_running_device(web_app) -> None:
    app, controller = web_app
    client = app.test_client()

    assert client.get("/api/cameras/right/snapshot.jpg").status_code == 401
    controller.session.state["logged_in"] = True
    assert client.get("/api/cameras/right/snapshot.jpg").status_code == 409


def test_d405_head_preview_reads_camera_shm(web_app, monkeypatch) -> None:
    app, controller = web_app
    controller.device_config_name = "i2rt_d405_yam_collect_async"
    controller.session.state["logged_in"] = True
    controller.processes.device.running = True
    opened: list[str] = []
    source_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    source_rgb[:, :] = (240, 30, 10)

    class FakeReader:
        def __init__(self, shm_name: str) -> None:
            opened.append(shm_name)

        def connect(self) -> None:
            pass

        def read_latest(self):
            return SimpleNamespace(
                timestamp_ns=time.time_ns(),
                data=source_rgb,
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "consumer.io.async_data_saver_i2rt_NEDF3.DeviceShmReader", FakeReader
    )
    client = app.test_client()

    response = client.get("/api/cameras/head/snapshot.jpg")

    assert response.status_code == 200
    assert opened == ["RealSenseCamera_4_data"]
    image = cv2.imdecode(np.frombuffer(response.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (480, 640, 3)
    mean_bgr = image.reshape(-1, 3).mean(axis=0)
    assert mean_bgr[2] > 220
    assert mean_bgr[0] < 30
    assert client.get("/api/state").get_json()["data"]["config"]["head_camera_configured"]
    assert 'class="camera-feed" data-camera-role="head"' in client.get("/").get_data(as_text=True)


def test_unknown_browser_asset_keeps_http_404(web_app) -> None:
    app, _controller = web_app

    response = app.test_client().get("/favicon.ico")

    assert response.status_code == 404


def test_collection_frontend_binds_calibration_state_before_rendering(web_app) -> None:
    del web_app
    source = (
        Path(__file__).parents[2]
        / "incubator"
        / "nedf3"
        / "I2RT_DM3"
        / "web"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")
    assert "const calibration = runtime.processes.calibration;" in source
    assert "byId(\"calibration-badge\")" in source


def test_collection_frontend_maps_only_requested_central_statuses(web_app) -> None:
    del web_app
    source = (
        Path(__file__).parents[2]
        / "incubator"
        / "nedf3"
        / "I2RT_DM3"
        / "web"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert 'return { label: "校准中", tone: "calibrating" };' in source
    assert source.count('return { label: "校准中", tone: "calibrating" };') >= 2
    assert 'return { label: "采集中", tone: "active" };' in source
    assert 'return { label: "遥操中", tone: "active" };' in source
    assert 'aligning: ["校准中", "warn"],' in source
    assert 'return { label: "对齐中", tone: "calibrating" };' not in source
    assert 'device.teleoperation_phase === "aligning"' in source
    assert "renderOperationStatus(null);" in source


def test_web_ui_keeps_desktop_layout_on_narrow_viewports(web_app) -> None:
    del web_app
    source = (
        Path(__file__).parents[2]
        / "incubator"
        / "nedf3"
        / "I2RT_DM3"
        / "web"
        / "static"
        / "app.css"
    ).read_text(encoding="utf-8")

    assert ".app-shell { min-width: 1180px;" in source
    assert "@media (max-width:" not in source


def test_calibration_logs_honor_incremental_offset(web_app) -> None:
    app, controller = web_app
    client = app.test_client()
    controller.processes.calibration.append_log("first calibration line\n")

    first = client.get("/api/logs?calibration=0").get_json()["data"][
        "calibration"
    ]
    assert first["lines"] == ["first calibration line\n"]

    controller.processes.calibration.append_log("second calibration line\n")
    second = client.get(
        f"/api/logs?calibration={first['offset']}"
    ).get_json()["data"]["calibration"]

    assert second["lines"] == ["second calibration line\n"]
    assert second["offset"] == first["offset"] + 1


def test_log_api_removes_terminal_color_sequences(web_app) -> None:
    app, controller = web_app
    controller.processes.device.append_log(
        "\x1b[32m2026-10-15 19:09:56.947\x1b[0m | "
        "\x1b[1mINFO\x1b[0m | stopped\n"
    )

    result = app.test_client().get("/api/logs?device=0").get_json()["data"]

    assert result["device"]["lines"] == [
        "2026-10-15 19:09:56.947 | INFO | stopped\n"
    ]


def test_control_posts_require_csrf_and_never_return_password(web_app) -> None:
    app, controller = web_app
    client = app.test_client()

    rejected = client.post(
        "/api/session/login",
        json={"phone": "13800000000", "password": "secret-value"},
    )
    assert rejected.status_code == 403

    accepted = client.post(
        "/api/session/login",
        json={"phone": "13800000000", "password": "secret-value"},
        headers=_csrf(app),
    )
    assert accepted.status_code == 200
    assert b"secret-value" not in accepted.data
    assert controller.upload_manager is not None
    assert controller.upload_manager.auto_enabled


def test_web_api_enforces_process_ordering(web_app) -> None:
    app, controller = web_app
    client = app.test_client()
    headers = _csrf(app)
    client.post(
        "/api/session/login",
        json={"phone": "13800000000", "password": "secret"},
        headers=headers,
    )
    tasks = client.get("/api/tasks?page=1&page_size=10").get_json()["data"]
    assert client.post(
        "/api/tasks/select", json={"index": -1}, headers=headers
    ).status_code == 400
    selected = client.post(
        "/api/tasks/select", json={"index": tasks[0]["index"]}, headers=headers
    )
    assert selected.status_code == 200
    assert selected.get_json()["data"]["session"]["task_selected"]
    assert selected.get_json()["data"]["config"]["resource_type"] == "i2rt"
    assert selected.get_json()["data"]["controls"]["can_calibrate_grippers"]
    assert controller.upload_manager is not None
    assert controller.upload_manager.configure_calls[-1]["resource_type"] == "i2rt"

    assert client.post("/api/device/start", json={}, headers=headers).status_code == 200
    assert client.post("/api/saver/start", json={}, headers=headers).status_code == 200
    blocked = client.post("/api/device/stop", json={}, headers=headers)
    assert blocked.status_code == 409
    assert controller.processes.device.running

    assert client.post("/api/saver/stop", json={}, headers=headers).status_code == 200
    assert client.post("/api/device/stop", json={}, headers=headers).status_code == 200


def test_visualizer_button_opens_latest_episode_only_when_idle(web_app) -> None:
    app, controller = web_app
    client = app.test_client()
    headers = _csrf(app)
    controller.session.state.update(logged_in=True, task_selected=True)
    state = client.get("/api/state").get_json()["data"]
    assert state["controls"]["can_visualize_episodes"]

    choices = client.get("/api/visualizer/episodes")
    assert choices.status_code == 200
    assert choices.get_json()["data"]["episodes"][0]["path"] == (
        "task/episode_0004"
    )

    controller.processes.saver.running = True
    blocked = client.post(
        "/api/visualizer/start",
        json={"episode": "task/episode_0004"},
        headers=headers,
    )
    assert blocked.status_code == 409
    controller.processes.saver.running = False

    opened = client.post(
        "/api/visualizer/start",
        json={"episode": "task/episode_0004"},
        headers=headers,
    )
    assert opened.status_code == 200
    assert opened.get_json()["data"]["episode"] == "episode_0004"
    assert controller.visualized_episodes == ["episode_0004"]


def test_hardware_profile_api_forwards_only_web_editable_bindings(
    web_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, controller = web_app
    client = app.test_client()
    headers = _csrf(app)
    controller.session.state["logged_in"] = True
    controller.selected_profile_id = "lktech_fisheye_2cam"
    profile = {
        "id": "lktech_fisheye_2cam",
        "revision": "revision-1",
        "display_name": "LKTech",
    }
    monkeypatch.setattr(controller, "system_profiles", lambda: [profile], raising=False)
    captured: dict[str, Any] = {}

    def save_binding(profile_id: str, **kwargs: Any) -> dict[str, Any]:
        captured.update(profile_id=profile_id, **kwargs)
        return {"profile_revision": kwargs["profile_revision"]}

    monkeypatch.setattr(
        controller,
        "save_hardware_binding",
        save_binding,
        raising=False,
    )

    listed = client.get("/api/device/profiles")
    assert listed.status_code == 200
    assert listed.get_json()["data"]["profiles"] == [profile]

    rejected = client.post(
        "/api/device/bindings",
        json={
            "profile_id": profile["id"],
            "profile_revision": profile["revision"],
            "cameras": {"left": "/dev/video0", "right": "/dev/video2"},
            "grippers": {"left": "/dev/ttyUSB0", "right": "/dev/ttyUSB1"},
            "can_roles": {"left": "can0"},
        },
        headers=headers,
    )
    assert rejected.status_code == 400

    accepted = client.post(
        "/api/device/bindings",
        json={
            "profile_id": profile["id"],
            "profile_revision": profile["revision"],
            "cameras": {"left": "/dev/video0", "right": "/dev/video2"},
            "grippers": {"left": "/dev/ttyUSB0", "right": "/dev/ttyUSB1"},
        },
        headers=headers,
    )
    assert accepted.status_code == 200
    assert captured["profile_id"] == profile["id"]
    assert "can_roles" not in captured


def test_gripper_calibration_requires_idle_processes(web_app) -> None:
    app, controller = web_app
    client = app.test_client()
    headers = _csrf(app)
    controller.session.state["logged_in"] = True
    no_task = client.post("/api/grippers/calibrate", json={}, headers=headers)
    assert no_task.status_code == 409
    controller.session.state["task_selected"] = True
    controller.processes.device.running = True
    blocked = client.post("/api/grippers/calibrate", json={}, headers=headers)
    assert blocked.status_code == 409
    controller.processes.device.running = False
    accepted = client.post("/api/grippers/calibrate", json={}, headers=headers)
    assert accepted.status_code == 200
    assert controller.processes.calibration.running
    assert not accepted.get_json()["data"]["controls"]["can_start_device"]

    device_start = client.post("/api/device/start", json={}, headers=headers)
    assert device_start.status_code == 409
    assert not controller.processes.device.running


def test_watchdog_fault_is_exposed_and_blocks_saver_start(web_app) -> None:
    app, controller = web_app
    client = app.test_client()
    controller.session.state.update(logged_in=True, task_selected=True)
    controller.processes.device.running = True
    controller.processes.device.ready_for_episode = True
    controller.processes.device.teleoperation_phase = "fault"
    controller.processes.device.teleoperation_fault = "left follower RPC timed out"

    payload = client.get("/api/state").get_json()["data"]

    assert payload["processes"]["device"]["teleoperation_phase"] == "fault"
    assert payload["processes"]["device"]["teleoperation_fault"] == (
        "left follower RPC timed out"
    )
    assert not payload["controls"]["can_start_saver"]
    assert not payload["controls"]["can_toggle_episode"]

    controller.processes.saver.running = True
    blocked = client.post("/api/episode/toggle", json={}, headers=_csrf(app))
    assert blocked.status_code == 409

    controller.processes.saver.recording = True
    payload = client.get("/api/state").get_json()["data"]
    assert payload["controls"]["can_toggle_episode"]
    assert payload["controls"]["can_discard_episode"]


def test_upload_settings_keep_resource_type_server_owned(web_app) -> None:
    app, controller = web_app
    client = app.test_client()
    headers = _csrf(app)
    client.post(
        "/api/session/login",
        json={"phone": "13800000000", "password": "secret"},
        headers=headers,
    )

    rejected = client.post(
        "/api/upload/settings",
        json={"resource_type": "should-not-be-accepted"},
        headers=headers,
    )
    assert rejected.status_code == 400

    accepted = client.post(
        "/api/upload/settings",
        json={"protect_minutes": 0, "upload_workers": 3},
        headers=headers,
    )
    assert accepted.status_code == 200
    assert controller.upload_manager is not None
    assert controller.upload_manager.configure_calls[-1]["resource_type"] == "i2rt"

    non_integer = client.post(
        "/api/upload/settings",
        json={"upload_workers": 2.5},
        headers=headers,
    )
    assert non_integer.status_code == 400


def test_task_output_path_cannot_escape_data_root() -> None:
    output = task_output_dir("../../ outside / task")
    chinese_output = task_output_dir("双臂装配任务")

    assert output.parent == task_output_dir("normal").parent
    assert ".." not in output.name
    assert "/" not in output.name
    assert chinese_output.name == "双臂装配任务"


def test_remote_binding_requires_explicit_override_contract() -> None:
    assert _is_loopback("127.0.0.1")
    assert _is_loopback("::1")
    assert _is_loopback("localhost")
    assert not _is_loopback("0.0.0.0")
