"""Backend shared by the I2RT DM3 login, collection, and upload tabs."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from devices.robots.i2rt_teleop.config import load_teleoperation_config
from incubator.nedf3.I2RT_DM3.process_manager import (
    DEFAULT_DEVICE_CONFIG,
    DEFAULT_SAVER_CONFIG,
    REPO_ROOT,
    I2RTProcessManager,
)
from incubator.nedf3.I2RT_DM3.hardware_bindings import (
    HardwareBindingStore,
    HardwareInventory,
    build_launch_plan,
    discover_hardware,
)
from incubator.nedf3.I2RT_DM3.system_profiles import SystemProfileRegistry
from incubator.nedf3.Vive_DM3.session import SessionStore
from incubator.nedf3.Vive_DM3.upload_manager import AutoUploadManager

DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "i2rt_nedf3"
I2RT_RESOURCE_TYPE = "i2rt"
D405_YAM_DEVICE_CONFIG = "i2rt_d405_yam_collect_async"
D405_YAM_TELEOP_CONFIG = REPO_ROOT / "configs" / "i2rt_d405_yam_teleop.yaml"


def task_output_dir(task_name: str) -> Path:
    """Return a task directory that cannot escape the managed data root."""
    safe_name = re.sub(r"[^\w.-]+", "_", task_name.strip(), flags=re.UNICODE)
    safe_name = safe_name.strip("._") or "i2rt_data_collection"
    return DEFAULT_DATA_ROOT / safe_name


def visualizable_episode_dirs(data_root: Path) -> list[Path]:
    """Return finalized NEDF3 episode folders below the managed data root."""
    root = data_root.expanduser().resolve()
    if not root.is_dir():
        return []
    episodes = [
        path.resolve()
        for path in root.rglob("episode_*")
        if path.is_dir()
        and path.resolve().is_relative_to(root)
        and not path.name.endswith(("_inprogress", "_temp"))
        and "quarantine" not in path.relative_to(root).parts
        and (path / "metadata.json").is_file()
        and (path / "data").is_dir()
    ]
    return sorted(episodes, key=lambda path: path.stat().st_mtime_ns, reverse=True)


def resolve_visualizable_episode(data_root: Path, relative_path: str) -> Path:
    """Resolve an operator selection without allowing escape from data root."""
    root = data_root.expanduser().resolve()
    selection = Path(relative_path)
    if selection.is_absolute() or not relative_path.strip():
        raise ValueError("Episode 路径必须是数据目录内的相对路径。")
    episode = (root / selection).resolve()
    if not episode.is_relative_to(root):
        raise ValueError("Episode 路径不能离开数据目录。")
    if episode not in visualizable_episode_dirs(root):
        raise ValueError("所选目录不是已完成的 NEDF3 Episode。")
    return episode


def validate_d405_yam_rig(
    environment: Optional[Mapping[str, str]] = None,
    teleoperation_config: Path = D405_YAM_TELEOP_CONFIG,
    camera_roles: tuple[str, ...] = ("left", "right", "head"),
) -> None:
    """Check identity bindings before starting any robot process."""
    env = dict(os.environ)
    if environment:
        env.update(environment)
    serial_environment = {
        "left": "SI_I2RT_LEFT_RS_SERIAL",
        "right": "SI_I2RT_RIGHT_RS_SERIAL",
        "head": "SI_I2RT_HEAD_RS_SERIAL",
    }
    roles = tuple(camera_roles)
    if not roles or not set(roles).issubset(serial_environment):
        raise RuntimeError("D405 相机角色配置无效。")
    serials = {
        role: env.get(serial_environment[role], "").strip() for role in roles
    }
    if not all(serials.values()) or len(set(serials.values())) != len(serials):
        if roles == ("left", "right", "head"):
            raise RuntimeError("请设置三个不同的左腕、右腕和头部 D405 序列号。")
        raise RuntimeError("请为当前方案设置不同的 D405 相机序列号。")

    confirmed_type = env.get("SI_I2RT_YAM_GRIPPER_TYPE", "").strip()
    teleoperation = load_teleoperation_config(teleoperation_config)
    configured_types = {
        arm.follower.gripper.i2rt_type for arm in teleoperation.arms.values()
    }
    if not confirmed_type or configured_types != {confirmed_type}:
        raise RuntimeError("请确认 YAM 夹爪型号与遥操作配置一致。")

    for side, arm in teleoperation.arms.items():
        for role in ("leader", "follower"):
            channel = getattr(arm, role).can_channel
            serial_env = f"SI_I2RT_CAN_{role.upper()}_{side.upper()}_SERIAL"
            expected_serial = env.get(serial_env, "").strip()
            if not expected_serial:
                raise RuntimeError(
                    f"请设置 {serial_env} 以确认 {channel} 的 CAN 适配器身份。"
                )
            try:
                properties = subprocess.run(
                    [
                        "udevadm",
                        "info",
                        "-q",
                        "property",
                        "-p",
                        f"/sys/class/net/{channel}",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=3,
                ).stdout
                link = json.loads(
                    subprocess.run(
                        ["ip", "-j", "-details", "link", "show", "dev", channel],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=3,
                    ).stdout
                )[0]
            except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
                raise RuntimeError(f"无法检查 {channel} CAN 接口：{exc}") from exc
            actual_serial = next(
                (
                    line.removeprefix("ID_SERIAL=")
                    for line in properties.splitlines()
                    if line.startswith("ID_SERIAL=")
                ),
                None,
            )
            if actual_serial != expected_serial:
                raise RuntimeError(f"{channel} CAN 适配器身份与 {serial_env} 不匹配。")
            bitrate = (
                link.get("linkinfo", {})
                .get("info_data", {})
                .get("bittiming", {})
                .get("bitrate")
            )
            if "UP" not in link.get("flags", []) or bitrate != 1_000_000:
                raise RuntimeError(
                    f"{channel} CAN 接口必须为 UP，波特率必须为 1 Mbit/s。"
                )

    try:
        import pyrealsense2 as rs

        connected = {
            device.get_info(rs.camera_info.serial_number)
            for device in rs.context().query_devices()
        }
    except Exception as exc:
        raise RuntimeError(f"无法检查 D405 相机：{exc}") from exc
    missing = set(serials.values()) - connected
    if missing:
        raise RuntimeError(f"找不到配置的 D405 相机：{', '.join(sorted(missing))}")


class I2RTDM3Controller:
    """Coordinate the DM3 session, I2RT processes, and upload manager."""

    def __init__(
        self,
        *,
        device_config_name: str = DEFAULT_DEVICE_CONFIG,
        saver_config_path: str = DEFAULT_SAVER_CONFIG,
        gripper_calibration_supported: bool = True,
        profile_registry: Optional[SystemProfileRegistry] = None,
        binding_store: Optional[HardwareBindingStore] = None,
        hardware_discovery: Callable[[], HardwareInventory] = discover_hardware,
    ) -> None:
        self.profile_registry = profile_registry or SystemProfileRegistry()
        self.binding_store = binding_store or HardwareBindingStore()
        self._hardware_discovery = hardware_discovery
        initial_profile = self.profile_registry.find_by_device_config(device_config_name)
        self.selected_profile_id = (
            initial_profile.profile_id
            if initial_profile is not None
            else self.profile_registry.default_profile_id
        )
        self.active_profile_id: Optional[str] = None
        self.session = SessionStore()
        self.processes = I2RTProcessManager(
            device_config_name=device_config_name,
            require_recording_profile=device_config_name == D405_YAM_DEVICE_CONFIG,
        )
        self.device_config_name = device_config_name
        self.saver_config_path = saver_config_path
        self.gripper_calibration_supported = (
            gripper_calibration_supported
            and device_config_name != D405_YAM_DEVICE_CONFIG
        )
        self.output_dir = str(DEFAULT_OUTPUT_DIR)
        self.task_name = "i2rt_data_collection"
        self.resource_type = I2RT_RESOURCE_TYPE
        self._upload_manager: Optional[AutoUploadManager] = None

    def system_profiles(self) -> list[dict[str, Any]]:
        return self.profile_registry.public_profiles()

    def select_system_profile(self, profile_id: str) -> dict[str, Any]:
        profile = self.profile_registry.get(profile_id)
        self.selected_profile_id = profile.profile_id
        self.device_config_name = profile.device_config
        self.saver_config_path = profile.saver_config
        self.gripper_calibration_supported = profile.calibration_supported
        self.processes.device_config_name = profile.device_config
        self.processes.require_recording_profile = profile.require_recording_profile
        return profile.public_dict()

    def hardware_inventory(self, profile_id: str) -> dict[str, Any]:
        profile = self.profile_registry.get(profile_id)
        inventory = self._hardware_discovery()
        available_backends = {camera.backend for camera in profile.cameras.values()}
        if profile.gripper_roles:
            available_backends.add("serial")
        public = inventory.public_dict()
        public["candidates"] = {
            backend: values
            for backend, values in public["candidates"].items()
            if backend in available_backends
        }
        public["profile"] = profile.public_dict()
        public["binding"] = self.binding_store.get(profile_id)
        return public

    def save_hardware_binding(
        self,
        profile_id: str,
        *,
        profile_revision: str,
        cameras: Mapping[str, Any],
        grippers: Mapping[str, Any],
    ) -> dict[str, Any]:
        profile = self.profile_registry.get(profile_id)
        if profile_revision != profile.revision:
            raise ValueError("系统方案已更新，请刷新页面后重新保存。")
        binding = self.binding_store.save(
            profile,
            cameras=cameras,
            grippers=grippers,
            inventory=self._hardware_discovery(),
        )
        self.select_system_profile(profile_id)
        return binding

    def apply_selected_task_defaults(self) -> None:
        state = self.session.public_state()
        task_name = str(state.get("task_name") or "i2rt_data_collection").strip()
        self.task_name = task_name
        self.output_dir = str(task_output_dir(task_name))
        self.processes.refresh_saved_episodes(self.output_dir)

    @property
    def upload_manager(self) -> Optional[AutoUploadManager]:
        return self._upload_manager

    def start_device(
        self,
        profile_id: Optional[str] = None,
        profile_revision: Optional[str] = None,
    ) -> None:
        if profile_id is None:
            if self.device_config_name == D405_YAM_DEVICE_CONFIG:
                validate_d405_yam_rig()
            self.processes.start_device(self.device_config_name)
            return

        profile = self.profile_registry.get(profile_id)
        if profile_revision != profile.revision:
            raise RuntimeError("系统方案版本已变化，请刷新硬件配置。")
        binding = self.binding_store.validate(profile, self._hardware_discovery())
        plan = build_launch_plan(profile, binding)
        if profile.device_config == D405_YAM_DEVICE_CONFIG:
            validate_d405_yam_rig(
                plan.environment,
                plan.teleop_config,
                tuple(profile.cameras),
            )
        self.select_system_profile(profile_id)
        self.processes.start_device(
            profile.device_config,
            environment=dict(plan.environment),
        )
        self.saver_config_path = str(plan.saver_config)
        self.active_profile_id = profile.profile_id

    def start_gripper_calibration(
        self,
        profile_id: Optional[str] = None,
        profile_revision: Optional[str] = None,
    ) -> None:
        if profile_id is None:
            if not self.gripper_calibration_supported:
                raise RuntimeError("当前夹爪不使用 LKTech 标定。")
            self.processes.start_calibration()
            return
        profile = self.profile_registry.get(profile_id)
        if not profile.calibration_supported:
            raise RuntimeError("当前夹爪不使用 LKTech 标定。")
        if profile_revision != profile.revision:
            raise RuntimeError("系统方案版本已变化，请刷新硬件配置。")
        binding = self.binding_store.validate(profile, self._hardware_discovery())
        plan = build_launch_plan(profile, binding)
        self.processes.start_calibration(str(plan.teleop_config))

    def start_saver(self) -> None:
        if not self.processes.device.ready_for_episode:
            raise RuntimeError("I2RT 设备仍在启动，请等待设备就绪。")
        session = self.session.require_ready_for_collection()
        metadata = session.metadata_for_saver()
        state = session.to_public_dict()
        self.processes.start_saver(
            config_path=self.saver_config_path,
            output_dir=self.output_dir,
            task_name=self.task_name,
            task_description=metadata["task_description"],
            dm3_user_id=metadata["user_id"],
            dm3_user_name=metadata["user_name"],
            task_execution_id=str(state.get("task_execution_id") or ""),
        )

    def visualizer_episodes(self) -> list[dict[str, Any]]:
        episodes = visualizable_episode_dirs(DEFAULT_DATA_ROOT)
        return [
            {
                "path": path.relative_to(DEFAULT_DATA_ROOT).as_posix(),
                "task": path.parent.relative_to(DEFAULT_DATA_ROOT).as_posix(),
                "name": path.name,
                "modified_at_ms": path.stat().st_mtime_ns // 1_000_000,
            }
            for path in episodes
        ]

    def start_visualizer(self, relative_path: str) -> Path:
        episode = resolve_visualizable_episode(DEFAULT_DATA_ROOT, relative_path)
        return self.processes.start_visualizer(episode)

    def create_upload_manager(
        self,
        data_dir: Path,
        log_callback=None,
    ) -> AutoUploadManager:
        """Create the upload queue for I2RT NEDF episodes."""

        def fetch_sts_token(_resource_type: str) -> dict[str, Any]:
            return self.session.get_client().fetch_oss_sts_token(
                resource_type=I2RT_RESOURCE_TYPE
            )

        resolved_dir = data_dir.expanduser().resolve()
        if self._upload_manager is None:
            self._upload_manager = AutoUploadManager(
                data_dir=resolved_dir,
                token_fetcher=fetch_sts_token,
                log_callback=log_callback,
            )
        else:
            self._upload_manager.configure(data_dir=resolved_dir)
            self._upload_manager.log_callback = log_callback
        self.resource_type = I2RT_RESOURCE_TYPE
        self._upload_manager.configure(resource_type=I2RT_RESOURCE_TYPE)
        return self._upload_manager

    def shutdown_upload_manager(self) -> None:
        if self._upload_manager is None:
            return
        self._upload_manager.shutdown()
        self._upload_manager = None

    def shutdown(self) -> None:
        self.processes.shutdown()
        self.shutdown_upload_manager()
        self.session.logout()


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_OUTPUT_DIR",
    "I2RT_RESOURCE_TYPE",
    "I2RTDM3Controller",
    "resolve_visualizable_episode",
    "task_output_dir",
    "visualizable_episode_dirs",
]
