"""Local Flask application for the I2RT DM3 collection workflow."""

from __future__ import annotations

import re
import secrets
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from incubator.nedf3.I2RT_DM3.controller import (
    DEFAULT_DATA_ROOT,
    D405_YAM_DEVICE_CONFIG,
    I2RT_RESOURCE_TYPE,
    I2RTDM3Controller,
)
from incubator.nedf3.Vive_DM3.upload_manager import AutoUploadManager
from incubator.nedf3.windows_DM3.dm3_api import DEFAULT_BASE_URL, DM3TaskExecution
from utils.logger_config import logger


MAX_REQUEST_BYTES = 32 * 1024
UPLOAD_ALL_CONFIRMATION = "UPLOAD_ALL"
CAMERA_PREVIEW_SHMS = {
    D405_YAM_DEVICE_CONFIG: {
        "right": "RealSenseCamera_2_data",
        "left": "RealSenseCamera_0_data",
        "head": "RealSenseCamera_4_data",
    },
    "default": {
        "right": "WebCameraDevice_2_data",
        "left": "WebCameraDevice_0_data",
    },
}
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class WebRequestError(RuntimeError):
    """Expected request failure returned to the browser as JSON."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class I2RTWebService:
    """Serialize browser commands around one process-owning controller."""

    def __init__(self, controller: Optional[I2RTDM3Controller] = None) -> None:
        self.controller = controller or I2RTDM3Controller()
        self._operation_lock = threading.RLock()
        self._upload_log_lock = threading.Lock()
        self._upload_logs: deque[str] = deque(maxlen=3000)
        self._upload_log_start = 0
        self._upload_log_next = 0
        self._tasks: list[DM3TaskExecution] = []
        self._shutdown = False
        self._stop_event = threading.Event()
        self._last_maintained_episode = ""
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            name="i2rt-dm3-web-maintenance",
            daemon=True,
        )
        self._maintenance_thread.start()

    def _append_upload_log(self, message: str) -> None:
        line = message.rstrip("\n") + "\n"
        with self._upload_log_lock:
            if len(self._upload_logs) == self._upload_logs.maxlen:
                self._upload_log_start += 1
            self._upload_logs.append(line)
            self._upload_log_next += 1

    def _ensure_upload_manager(self) -> AutoUploadManager:
        manager = self.controller.upload_manager
        if manager is None:
            manager = self.controller.create_upload_manager(
                DEFAULT_DATA_ROOT,
                log_callback=self._append_upload_log,
            )
        else:
            manager.log_callback = self._append_upload_log
        return manager

    def state(self) -> dict[str, Any]:
        session = self.controller.session.public_state()
        processes = self.controller.processes.runtime_status()
        device = processes["device"]
        saver = processes["saver"]
        calibration = processes["calibration"]
        visualizer = processes.get(
            "visualizer",
            {"running": False, "stopping": False},
        )
        device_live = bool(device["running"] or device["stopping"])
        saver_live = bool(saver["running"] or saver["stopping"])
        calibration_live = bool(calibration["running"] or calibration["stopping"])
        visualizer_live = bool(visualizer["running"] or visualizer["stopping"])
        teleoperation_faulted = device["teleoperation_phase"] == "fault"
        logged_in = bool(session["logged_in"])
        manager = self.controller.upload_manager
        upload = (
            manager.upload_progress_snapshot()
            if manager is not None
            else {
                "auto_enabled": False,
                "worker_running": False,
                "current_episode": None,
                "queued_count": 0,
                "performance": {"running": False, "alerts": []},
            }
        )
        profile_registry = getattr(self.controller, "profile_registry", None)
        selected_profile_id = getattr(self.controller, "selected_profile_id", None)
        if profile_registry is not None and selected_profile_id:
            head_camera_configured = "head" in profile_registry.get(
                selected_profile_id
            ).cameras
        else:
            head_camera_configured = (
                self.controller.device_config_name == D405_YAM_DEVICE_CONFIG
            )
        return {
            "session": session,
            "config": {
                "device": self.controller.device_config_name,
                "saver": self.controller.saver_config_path,
                "output_dir": self.controller.output_dir,
                "resource_type": I2RT_RESOURCE_TYPE,
                "head_camera_configured": head_camera_configured,
                "dm3_base_url": DEFAULT_BASE_URL,
                "system_profile": getattr(
                    self.controller, "selected_profile_id", None
                ),
                "active_system_profile": getattr(
                    self.controller, "active_profile_id", None
                ),
            },
            "processes": processes,
            "upload": upload,
            "controls": {
                "calibration_supported": getattr(
                    self.controller, "gripper_calibration_supported", True
                ),
                "can_start_device": (
                    logged_in and not device_live and not calibration_live
                ),
                "can_stop_device": device_live and not saver_live,
                "can_start_saver": (
                    logged_in
                    and bool(session["task_selected"])
                    and bool(device["ready_for_episode"])
                    and not teleoperation_faulted
                    and not saver_live
                ),
                "can_stop_saver": saver_live and not saver["stopping"],
                "can_toggle_episode": (
                    bool(saver["running"])
                    and not bool(saver["finalizing_episode"])
                    and (bool(saver["recording"]) or not teleoperation_faulted)
                ),
                "can_discard_episode": (
                    bool(saver["running"])
                    and bool(saver["recording"])
                    and not bool(saver["finalizing_episode"])
                ),
                "can_calibrate_grippers": (
                    getattr(self.controller, "gripper_calibration_supported", True)
                    and logged_in
                    and bool(session["task_selected"])
                    and not device_live
                    and not saver_live
                    and not calibration_live
                ),
                "can_visualize_episodes": (
                    logged_in
                    and bool(session["task_selected"])
                    and not device_live
                    and not saver_live
                    and not calibration_live
                    and not visualizer_live
                ),
            },
        }

    def login(self, phone: str, password: str) -> dict[str, Any]:
        phone = phone.strip()
        if not phone or not password:
            raise WebRequestError("请输入手机号和密码。")
        with self._operation_lock:
            if self.controller.session.public_state()["logged_in"]:
                raise WebRequestError("当前已有登录会话。", 409)
            self.controller.session.login(phone, password, DEFAULT_BASE_URL)
            manager = self._ensure_upload_manager()
            manager.start_auto()
            self._append_upload_log("DM3 登录成功，自动上传已启动")
            return self.state()

    def logout(self) -> dict[str, Any]:
        with self._operation_lock:
            runtime = self.controller.processes.runtime_status()
            if runtime["device"]["running"] or runtime["saver"]["running"]:
                raise WebRequestError("停止保存与设备后才能退出登录。", 409)
            self.controller.shutdown_upload_manager()
            self.controller.session.logout()
            self._tasks = []
            self._last_maintained_episode = ""
            return self.state()

    def fetch_tasks(
        self,
        *,
        page: int,
        page_size: int,
        execute_day: Optional[str],
    ) -> list[dict[str, Any]]:
        if page < 1 or page_size < 1 or page_size > 100:
            raise WebRequestError("页码需大于 0，每页数量需为 1 到 100。")
        if execute_day and not _DATE_PATTERN.fullmatch(execute_day):
            raise WebRequestError("执行日期必须使用 YYYY-MM-DD 格式。")
        with self._operation_lock:
            self._tasks = self.controller.session.fetch_tasks(
                page=page,
                page_size=page_size,
                execute_day=execute_day,
            )
            return [self._task_dict(index, task) for index, task in enumerate(self._tasks)]

    def select_task(self, index: int) -> dict[str, Any]:
        if index < 0:
            raise WebRequestError("任务索引不能小于 0。")
        with self._operation_lock:
            if self.controller.processes.saver.running:
                raise WebRequestError("保存进程运行时不能切换任务。", 409)
            self.controller.session.select_task_by_index(index)
            self.controller.apply_selected_task_defaults()
            manager = self.controller.upload_manager
            if manager is not None:
                manager.configure(
                    data_dir=DEFAULT_DATA_ROOT, resource_type=I2RT_RESOURCE_TYPE
                )
            return self.state()

    @staticmethod
    def _task_dict(index: int, task: DM3TaskExecution) -> dict[str, Any]:
        return {
            "index": index,
            "id": task.id,
            "name": task.name,
            "status": task.status,
        }

    def system_profiles(self) -> dict[str, Any]:
        profiles = getattr(self.controller, "system_profiles", None)
        if profiles is None:
            return {"profiles": [], "selected_profile_id": None}
        return {
            "profiles": profiles(),
            "selected_profile_id": self.controller.selected_profile_id,
        }

    def _require_hardware_idle(self) -> None:
        runtime = self.controller.processes.runtime_status()
        if any(
            runtime[name][key]
            for name in ("device", "saver", "calibration")
            for key in ("running", "stopping")
        ):
            raise WebRequestError("停止设备、保存和标定进程后才能修改硬件配置。", 409)

    def select_system_profile(self, profile_id: str) -> dict[str, Any]:
        self._require_login()
        with self._operation_lock:
            self._require_hardware_idle()
            self.controller.select_system_profile(profile_id)
            return self.state()

    def hardware_inventory(self, profile_id: str) -> dict[str, Any]:
        self._require_login()
        with self._operation_lock:
            return self.controller.hardware_inventory(profile_id)

    def save_hardware_binding(
        self,
        profile_id: str,
        profile_revision: str,
        cameras: Mapping[str, Any],
        grippers: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._require_login()
        with self._operation_lock:
            self._require_hardware_idle()
            binding = self.controller.save_hardware_binding(
                profile_id,
                profile_revision=profile_revision,
                cameras=cameras,
                grippers=grippers,
            )
            return {"binding": binding, "state": self.state()}

    def start_device(
        self,
        profile_id: Optional[str] = None,
        profile_revision: Optional[str] = None,
    ) -> dict[str, Any]:
        with self._operation_lock:
            if not self.controller.session.public_state()["logged_in"]:
                raise WebRequestError("请先登录 DM3。", 401)
            calibration = self.controller.processes.calibration
            if (
                calibration.running
                or calibration.stopping
                or calibration.process is not None
            ):
                raise WebRequestError("夹爪标定进程运行时不能启动设备。", 409)
            if profile_id is None:
                self.controller.start_device()
            else:
                self.controller.start_device(profile_id, profile_revision)
            return self.state()

    def _preview_shms(self) -> Mapping[str, str]:
        registry = getattr(self.controller, "profile_registry", None)
        profile_id = getattr(self.controller, "active_profile_id", None) or getattr(
            self.controller, "selected_profile_id", None
        )
        if registry is not None and profile_id:
            profile = registry.get(profile_id)
            return {
                role: camera.preview_shm for role, camera in profile.cameras.items()
            }
        return CAMERA_PREVIEW_SHMS.get(
            self.controller.device_config_name,
            CAMERA_PREVIEW_SHMS["default"],
        )

    def stop_device(self) -> dict[str, Any]:
        with self._operation_lock:
            saver = self.controller.processes.saver
            if saver.running or saver.stopping or saver.process is not None:
                raise WebRequestError("停止设备前请先停止 NEDF3 保存进程。", 409)
            self.controller.processes.stop_device()
            return self.state()

    def start_saver(self) -> dict[str, Any]:
        with self._operation_lock:
            self.controller.start_saver()
            return self.state()

    def stop_saver(self) -> dict[str, Any]:
        with self._operation_lock:
            self.controller.processes.stop_saver()
            return self.state()

    def visualizer_episodes(self) -> dict[str, Any]:
        self._require_login()
        return {"episodes": self.controller.visualizer_episodes()}

    def start_visualizer(self, relative_path: str) -> dict[str, Any]:
        self._require_login()
        with self._operation_lock:
            if not self.controller.session.public_state()["task_selected"]:
                raise WebRequestError("请先选择 DM3 任务。", 409)
            runtime = self.controller.processes.runtime_status()
            if any(
                runtime[name][key]
                for name in ("device", "saver", "calibration", "visualizer")
                for key in ("running", "stopping")
            ):
                raise WebRequestError(
                    "启动 Rerun 前请停止设备、保存和标定进程。",
                    409,
                )
            episode_dir = self.controller.start_visualizer(relative_path)
            return {"episode": episode_dir.name, "state": self.state()}

    def calibrate_grippers(
        self,
        profile_id: Optional[str] = None,
        profile_revision: Optional[str] = None,
    ) -> dict[str, Any]:
        with self._operation_lock:
            self._require_login()
            if not self.controller.session.public_state()["task_selected"]:
                raise WebRequestError("请先选择 DM3 任务。", 409)
            runtime = self.controller.processes.runtime_status()
            if any(
                runtime[name][key]
                for name in ("device", "saver", "calibration")
                for key in ("running", "stopping")
            ):
                raise WebRequestError("标定前请停止设备、保存和已有标定进程。", 409)
            if profile_id is None:
                self.controller.start_gripper_calibration()
            else:
                self.controller.start_gripper_calibration(
                    profile_id, profile_revision
                )
            return self.state()

    def episode_command(self, command: str) -> dict[str, Any]:
        if command not in {"toggle", "discard"}:
            raise WebRequestError("不支持的 Episode 命令。")
        with self._operation_lock:
            saver = self.controller.processes.saver
            if saver.finalizing_episode:
                raise WebRequestError("Episode 正在完成，请稍候。", 409)
            if (
                command == "toggle"
                and not saver.recording
                and self.controller.processes.device.teleoperation_phase == "fault"
            ):
                raise WebRequestError("遥操作故障后不能开始新的 Episode。", 409)
            if command == "discard" and not saver.recording:
                raise WebRequestError("当前没有正在采集的 Episode。", 409)
            self.controller.processes.send_saver_command(command)
            return self.state()

    def upload_snapshot(self, *, refresh: bool = False) -> dict[str, Any]:
        self._require_login()
        manager = self._ensure_upload_manager()
        if refresh:
            manager.scan()
        return manager.snapshot(background_refresh=not refresh)

    def configure_upload(self, values: Mapping[str, Any]) -> dict[str, Any]:
        self._require_login()
        bounds = {
            "protect_minutes": (0, 24 * 60),
            "keep_latest_count": (0, 1000),
            "max_retries": (0, 20),
            "upload_workers": (1, 16),
            "multipart_threads": (1, 32),
        }
        parsed: dict[str, int] = {}
        for name, (minimum, maximum) in bounds.items():
            if name not in values:
                continue
            value = _strict_int(values[name], name)
            if not minimum <= value <= maximum:
                raise WebRequestError(f"{name} 必须在 {minimum} 到 {maximum} 之间。")
            parsed[name] = value
        if not parsed:
            raise WebRequestError("没有可更新的上传参数。")
        with self._operation_lock:
            manager = self._ensure_upload_manager()
            manager.configure(resource_type=I2RT_RESOURCE_TYPE, **parsed)
            return manager.snapshot()

    def set_auto_upload(self, enabled: bool) -> dict[str, Any]:
        self._require_login()
        with self._operation_lock:
            manager = self._ensure_upload_manager()
            if enabled:
                manager.start_auto()
            else:
                manager.stop_auto()
            return manager.upload_progress_snapshot()

    def enqueue_uploads(self, mode: str, confirmation: str = "") -> dict[str, Any]:
        self._require_login()
        with self._operation_lock:
            manager = self._ensure_upload_manager()
            if mode == "eligible":
                count = manager.enqueue_eligible(force=False)
            elif mode == "failed":
                count = manager.enqueue_failed()
            elif mode == "all":
                if confirmation != UPLOAD_ALL_CONFIRMATION:
                    raise WebRequestError("确认全部上传需要明确确认。", 409)
                count = manager.enqueue_eligible(force=True)
            else:
                raise WebRequestError("不支持的上传入队模式。")
            return {"ok": True, "queued": count}

    def logs(self, offsets: Mapping[str, int]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name, managed in (
            ("device", self.controller.processes.device),
            ("saver", self.controller.processes.saver),
            ("calibration", self.controller.processes.calibration),
        ):
            lines, next_offset = managed.read_logs_since(offsets.get(name, 0))
            payload[name] = {"lines": lines, "offset": next_offset}
        with self._upload_log_lock:
            offset = max(offsets.get("upload", 0), self._upload_log_start)
            start = max(0, offset - self._upload_log_start)
            lines = list(self._upload_logs)[start:]
            payload["upload"] = {"lines": lines, "offset": self._upload_log_next}
        return payload

    def camera_snapshot(self, role: str) -> bytes:
        """Encode the latest shared-memory camera frame for the local UI."""
        if role not in {"right", "left", "head"}:
            raise WebRequestError("未知相机。", 404)
        self._require_login()
        if not self.controller.processes.device.running:
            raise WebRequestError("设备未运行。", 409)

        import cv2
        import numpy as np

        from consumer.io.async_data_saver_i2rt_NEDF3 import (
            DeviceFrame,
            DeviceShmReader,
        )

        preview_shms = self._preview_shms()
        if role not in preview_shms:
            frame = DeviceFrame(time.time_ns(), np.zeros((480, 640, 3), dtype=np.uint8))
        else:
            reader = DeviceShmReader(preview_shms[role])
            try:
                reader.connect()
                frame = reader.read_latest()
            finally:
                reader.close()
            if frame is None or time.time_ns() - frame.timestamp_ns > 2_000_000_000:
                raise WebRequestError("相机画面不可用。", 409)

        image_bgr = frame.data[:, :, ::-1]
        ok, encoded = cv2.imencode(
            ".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        if not ok:
            raise RuntimeError("相机预览编码失败。")
        return encoded.tobytes()

    def camera_stream(self, role: str) -> Iterator[bytes]:
        """Yield a latest-frame MJPEG stream without reopening the camera."""
        if role not in {"right", "left", "head"}:
            raise WebRequestError("未知相机。", 404)
        self._require_login()
        if not self.controller.processes.device.running:
            raise WebRequestError("设备未运行。", 409)

        import cv2
        import numpy as np

        from consumer.io.async_data_saver_i2rt_NEDF3 import DeviceShmReader

        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        preview_shms = self._preview_shms()
        if role not in preview_shms:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            ok, encoded = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            if not ok:
                raise RuntimeError("相机预览编码失败。")
            payload = boundary + encoded.tobytes() + b"\r\n"

            def synthetic_stream() -> Iterator[bytes]:
                while self.controller.processes.device.running:
                    yield payload
                    time.sleep(0.1)

            return synthetic_stream()

        reader = DeviceShmReader(preview_shms[role])

        def hardware_stream() -> Iterator[bytes]:
            last_timestamp_ns = 0
            next_frame_time = time.monotonic()
            try:
                reader.connect()
                while self.controller.processes.device.running:
                    frame = reader.read_latest()
                    if frame is None or frame.timestamp_ns <= last_timestamp_ns:
                        time.sleep(0.002)
                        continue
                    now = time.monotonic()
                    if now < next_frame_time:
                        time.sleep(next_frame_time - now)
                    image_bgr = frame.data[:, :, ::-1]
                    ok, encoded = cv2.imencode(
                        ".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90]
                    )
                    if not ok:
                        continue
                    last_timestamp_ns = frame.timestamp_ns
                    next_frame_time = time.monotonic() + 1.0 / 30.0
                    yield boundary + encoded.tobytes() + b"\r\n"
            finally:
                reader.close()

        return hardware_stream()

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._stop_event.set()
        self._maintenance_thread.join(timeout=2.0)
        with self._operation_lock:
            self.controller.shutdown()

    def _require_login(self) -> None:
        if not self.controller.session.public_state()["logged_in"]:
            raise WebRequestError("请先登录 DM3。", 401)

    def _maintenance_loop(self) -> None:
        while not self._stop_event.wait(0.5):
            try:
                self._maintain_finalized_episode()
            except Exception as exc:
                logger.warning("I2RT DM3 web upload maintenance failed: %s", exc)
                self._append_upload_log(f"自动上传刷新失败：{exc}")

    def _maintain_finalized_episode(self) -> None:
        manager = self.controller.upload_manager
        saver = self.controller.processes.saver
        episode_name = saver.last_episode_name
        if manager is None or episode_name in {"", "—", self._last_maintained_episode}:
            return
        episode_path = Path(self.controller.output_dir) / episode_name
        result = manager.refresh_after_episode(episode_path)
        queued = 0
        if manager.upload_progress_snapshot()["auto_enabled"]:
            queued = manager.enqueue_eligible_episodes(result.episodes, force=False)
        self._last_maintained_episode = episode_name
        self._append_upload_log(
            f"Episode 刷新完成：{episode_name}，自动入队 {queued} 条"
        )


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise WebRequestError(f"{name} 必须是整数。")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value)
    raise WebRequestError(f"{name} 必须是整数。")


def _json_body(*, allowed: set[str], required: set[str] = frozenset()) -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise WebRequestError("请求正文必须是 JSON 对象。")
    unexpected = set(payload) - allowed
    missing = required - set(payload)
    if unexpected:
        raise WebRequestError(f"包含不支持的字段：{', '.join(sorted(unexpected))}")
    if missing:
        raise WebRequestError(f"缺少字段：{', '.join(sorted(missing))}")
    return payload


def create_app(
    controller: Optional[I2RTDM3Controller] = None,
    *,
    service: Optional[I2RTWebService] = None,
) -> Flask:
    """Construct the hardware-owning web application."""
    package_root = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(package_root / "templates"),
        static_folder=str(package_root / "static"),
        static_url_path="/static",
    )
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_REQUEST_BYTES,
        I2RT_CSRF_TOKEN=secrets.token_urlsafe(32),
    )
    web_service = service or I2RTWebService(controller)
    app.config["I2RT_WEB_SERVICE"] = web_service

    @app.after_request
    def secure_response(response: Response) -> Response:
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        )
        return response

    @app.before_request
    def verify_control_request() -> Optional[Response]:
        if request.method == "POST" and request.headers.get("X-I2RT-CSRF") != app.config["I2RT_CSRF_TOKEN"]:
            raise WebRequestError("控制请求校验失败。", 403)
        return None

    @app.errorhandler(WebRequestError)
    def expected_error(exc: WebRequestError) -> tuple[Response, int]:
        return jsonify({"ok": False, "error": str(exc)}), exc.status_code

    @app.errorhandler(RuntimeError)
    @app.errorhandler(ValueError)
    def operation_error(exc: Exception) -> tuple[Response, int]:
        return jsonify({"ok": False, "error": str(exc)}), 409

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException) -> HTTPException:
        return exc

    @app.errorhandler(Exception)
    def unexpected_error(exc: Exception) -> tuple[Response, int]:
        logger.exception("Unhandled I2RT DM3 web request error")
        return jsonify({"ok": False, "error": "服务器操作失败，请查看终端日志。"}), 500

    def command(operation: Callable[[], Any]) -> Response:
        return jsonify({"ok": True, "data": operation()})

    @app.get("/")
    def index() -> Response:
        profile_registry = getattr(web_service.controller, "profile_registry", None)
        selected_profile_id = getattr(
            web_service.controller, "selected_profile_id", None
        )
        if profile_registry is not None and selected_profile_id:
            head_camera_configured = "head" in profile_registry.get(
                selected_profile_id
            ).cameras
        else:
            head_camera_configured = (
                web_service.controller.device_config_name == D405_YAM_DEVICE_CONFIG
            )
        return app.make_response(
            render_template(
                "index.html",
                csrf_token=app.config["I2RT_CSRF_TOKEN"],
                dm3_base_url=DEFAULT_BASE_URL,
                head_camera_configured=head_camera_configured,
            )
        )

    @app.get("/api/health")
    def health() -> Response:
        return jsonify({"ok": True, "service": "i2rt-dm3"})

    @app.get("/api/state")
    def state() -> Response:
        return jsonify({"ok": True, "data": web_service.state()})

    @app.get("/api/device/profiles")
    def system_profiles() -> Response:
        return jsonify({"ok": True, "data": web_service.system_profiles()})

    @app.get("/api/device/inventory")
    def hardware_inventory() -> Response:
        profile_id = request.args.get("profile_id", "").strip()
        if not profile_id:
            raise WebRequestError("缺少 profile_id。")
        return jsonify({"ok": True, "data": web_service.hardware_inventory(profile_id)})

    @app.post("/api/device/profile")
    def select_system_profile() -> Response:
        payload = _json_body(allowed={"profile_id"}, required={"profile_id"})
        if not isinstance(payload["profile_id"], str):
            raise WebRequestError("profile_id 必须是字符串。")
        return command(lambda: web_service.select_system_profile(payload["profile_id"]))

    @app.post("/api/device/bindings")
    def save_hardware_binding() -> Response:
        payload = _json_body(
            allowed={"profile_id", "profile_revision", "cameras", "grippers"},
            required={"profile_id", "profile_revision", "cameras", "grippers"},
        )
        if not all(
            isinstance(payload[name], str)
            for name in ("profile_id", "profile_revision")
        ) or not all(isinstance(payload[name], dict) for name in ("cameras", "grippers")):
            raise WebRequestError("硬件绑定请求格式无效。")
        return command(
            lambda: web_service.save_hardware_binding(
                payload["profile_id"],
                payload["profile_revision"],
                payload["cameras"],
                payload["grippers"],
            )
        )

    @app.post("/api/session/login")
    def login() -> Response:
        payload = _json_body(allowed={"phone", "password"}, required={"phone", "password"})
        if not isinstance(payload["phone"], str) or not isinstance(payload["password"], str):
            raise WebRequestError("手机号和密码必须是字符串。")
        return command(lambda: web_service.login(payload["phone"], payload["password"]))

    @app.post("/api/session/logout")
    def logout() -> Response:
        _json_body(allowed=set())
        return command(web_service.logout)

    @app.get("/api/tasks")
    def tasks() -> Response:
        page = _strict_int(request.args.get("page", 1), "page")
        page_size = _strict_int(request.args.get("page_size", 10), "page_size")
        execute_day = request.args.get("execute_day", "").strip() or None
        return jsonify(
            {
                "ok": True,
                "data": web_service.fetch_tasks(
                    page=page,
                    page_size=page_size,
                    execute_day=execute_day,
                ),
            }
        )

    @app.post("/api/tasks/select")
    def select_task() -> Response:
        payload = _json_body(allowed={"index"}, required={"index"})
        return command(lambda: web_service.select_task(_strict_int(payload["index"], "index")))

    @app.post("/api/device/start")
    def start_device() -> Response:
        payload = _json_body(allowed={"profile_id", "profile_revision"})
        if not payload:
            return command(web_service.start_device)
        if set(payload) != {"profile_id", "profile_revision"} or not all(
            isinstance(payload[name], str)
            for name in ("profile_id", "profile_revision")
        ):
            raise WebRequestError("启动请求必须同时包含系统方案及其版本。")
        return command(
            lambda: web_service.start_device(
                payload["profile_id"], payload["profile_revision"]
            )
        )

    @app.post("/api/device/stop")
    def stop_device() -> Response:
        _json_body(allowed=set())
        return command(web_service.stop_device)

    @app.post("/api/saver/start")
    def start_saver() -> Response:
        _json_body(allowed=set())
        return command(web_service.start_saver)

    @app.post("/api/saver/stop")
    def stop_saver() -> Response:
        _json_body(allowed=set())
        return command(web_service.stop_saver)

    @app.get("/api/visualizer/episodes")
    def visualizer_episodes() -> Response:
        return jsonify({"ok": True, "data": web_service.visualizer_episodes()})

    @app.post("/api/visualizer/start")
    def start_visualizer() -> Response:
        payload = _json_body(allowed={"episode"}, required={"episode"})
        if not isinstance(payload["episode"], str):
            raise WebRequestError("Episode 路径必须是字符串。")
        return command(lambda: web_service.start_visualizer(payload["episode"]))

    @app.post("/api/grippers/calibrate")
    def calibrate_grippers() -> Response:
        payload = _json_body(allowed={"profile_id", "profile_revision"})
        if not payload:
            return command(web_service.calibrate_grippers)
        if set(payload) != {"profile_id", "profile_revision"} or not all(
            isinstance(payload[name], str)
            for name in ("profile_id", "profile_revision")
        ):
            raise WebRequestError("标定请求必须同时包含系统方案及其版本。")
        return command(
            lambda: web_service.calibrate_grippers(
                payload["profile_id"], payload["profile_revision"]
            )
        )

    @app.post("/api/episode/<command_name>")
    def episode(command_name: str) -> Response:
        _json_body(allowed=set())
        return command(lambda: web_service.episode_command(command_name))

    @app.get("/api/logs")
    def logs() -> Response:
        offsets = {
            name: max(0, _strict_int(request.args.get(name, 0), name))
            for name in ("device", "saver", "calibration", "upload")
        }
        return jsonify({"ok": True, "data": web_service.logs(offsets)})

    @app.get("/api/cameras/<role>/snapshot.jpg")
    def camera_snapshot(role: str) -> Response:
        return Response(web_service.camera_snapshot(role), mimetype="image/jpeg")

    @app.get("/api/cameras/<role>/stream.mjpg")
    def camera_stream(role: str) -> Response:
        stream = web_service.camera_stream(role)
        return Response(
            stream,
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/upload")
    def upload() -> Response:
        return jsonify({"ok": True, "data": web_service.upload_snapshot()})

    @app.post("/api/upload/refresh")
    def refresh_upload() -> Response:
        _json_body(allowed=set())
        return command(lambda: web_service.upload_snapshot(refresh=True))

    @app.post("/api/upload/settings")
    def upload_settings() -> Response:
        allowed = {
            "protect_minutes",
            "keep_latest_count",
            "max_retries",
            "upload_workers",
            "multipart_threads",
        }
        payload = _json_body(allowed=allowed)
        return command(lambda: web_service.configure_upload(payload))

    @app.post("/api/upload/auto")
    def upload_auto() -> Response:
        payload = _json_body(allowed={"enabled"}, required={"enabled"})
        if not isinstance(payload["enabled"], bool):
            raise WebRequestError("enabled 必须是布尔值。")
        return command(lambda: web_service.set_auto_upload(payload["enabled"]))

    @app.post("/api/upload/enqueue")
    def upload_enqueue() -> Response:
        payload = _json_body(allowed={"mode", "confirmation"}, required={"mode"})
        return command(
            lambda: web_service.enqueue_uploads(
                str(payload["mode"]),
                str(payload.get("confirmation") or ""),
            )
        )

    return app


__all__ = [
    "I2RTWebService",
    "UPLOAD_ALL_CONFIRMATION",
    "WebRequestError",
    "create_app",
]
