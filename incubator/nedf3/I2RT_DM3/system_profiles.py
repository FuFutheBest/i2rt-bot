"""Backend-owned I2RT system profile manifest."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from incubator.nedf3.I2RT_DM3.process_manager import REPO_ROOT

DEFAULT_PROFILE_MANIFEST = REPO_ROOT / "configs" / "i2rt_system_profiles.yaml"
_PROFILE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_CAMERA_BACKENDS = {"v4l2", "realsense"}
_CAMERA_ROLES = {"left", "right", "head"}
_GRIPPER_BACKENDS = {"lktech", "i2rt_can"}


@dataclass(frozen=True)
class CameraProfile:
    role: str
    backend: str
    environment: str
    preview_shm: str


@dataclass(frozen=True)
class SystemProfile:
    profile_id: str
    display_name: str
    device_config: str
    teleop_config: Path
    saver_config: str
    require_recording_profile: bool
    gripper_backend: str
    gripper_roles: tuple[str, ...]
    calibration_supported: bool
    cameras: Mapping[str, CameraProfile]
    revision: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.profile_id,
            "display_name": self.display_name,
            "revision": self.revision,
            "device_config": self.device_config,
            "gripper": {
                "backend": self.gripper_backend,
                "calibration_supported": self.calibration_supported,
                "roles": list(self.gripper_roles),
            },
            "cameras": {
                role: {"backend": camera.backend}
                for role, camera in self.cameras.items()
            },
        }


class SystemProfileRegistry:
    """Load and validate the finite set of Web-selectable system profiles."""

    def __init__(self, manifest_path: Path = DEFAULT_PROFILE_MANIFEST) -> None:
        self.manifest_path = manifest_path.resolve()
        raw = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("I2RT 系统方案清单版本无效。")
        profiles_raw = raw.get("profiles")
        if not isinstance(profiles_raw, dict) or not profiles_raw:
            raise ValueError("I2RT 系统方案清单不能为空。")
        self._profiles = {
            profile_id: self._parse_profile(profile_id, body)
            for profile_id, body in profiles_raw.items()
        }
        self.default_profile_id = str(raw.get("default_profile") or "")
        if self.default_profile_id not in self._profiles:
            raise ValueError("I2RT 默认系统方案不存在。")

    def _repo_file(self, value: Any, field: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"系统方案的 {field} 无效。")
        candidate = (REPO_ROOT / value).resolve()
        try:
            candidate.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"系统方案的 {field} 必须位于仓库内。") from exc
        if not candidate.is_file():
            raise ValueError(f"系统方案引用的文件不存在：{value}")
        return candidate

    def _parse_profile(self, profile_id: Any, body: Any) -> SystemProfile:
        if not isinstance(profile_id, str) or not _PROFILE_ID_RE.fullmatch(profile_id):
            raise ValueError(f"系统方案 ID 无效：{profile_id}")
        if not isinstance(body, dict):
            raise ValueError(f"系统方案 {profile_id} 必须是对象。")
        cameras_raw = body.get("cameras")
        if (
            not isinstance(cameras_raw, dict)
            or not {"left", "right"}.issubset(cameras_raw)
            or not set(cameras_raw).issubset(_CAMERA_ROLES)
        ):
            raise ValueError(
                f"系统方案 {profile_id} 必须定义左右腕部相机，可选头部相机。"
            )
        cameras: dict[str, CameraProfile] = {}
        for role, camera_raw in cameras_raw.items():
            if not isinstance(camera_raw, dict):
                raise ValueError(f"系统方案 {profile_id} 的相机定义无效。")
            backend = str(camera_raw.get("backend") or "")
            if backend not in _CAMERA_BACKENDS:
                raise ValueError(f"系统方案 {profile_id} 的相机后端无效。")
            environment = str(camera_raw.get("environment") or "")
            preview_shm = str(camera_raw.get("preview_shm") or "")
            if not environment.startswith("SI_I2RT_") or not preview_shm:
                raise ValueError(f"系统方案 {profile_id} 的相机契约无效。")
            cameras[role] = CameraProfile(role, backend, environment, preview_shm)

        gripper = body.get("gripper")
        if not isinstance(gripper, dict):
            raise ValueError(f"系统方案 {profile_id} 缺少夹爪定义。")
        backend = str(gripper.get("backend") or "")
        roles = tuple(str(role) for role in gripper.get("roles", []))
        if backend not in _GRIPPER_BACKENDS:
            raise ValueError(f"系统方案 {profile_id} 的夹爪后端无效。")
        if backend == "lktech" and set(roles) != {"left", "right"}:
            raise ValueError("LKTech 系统方案必须定义左右夹爪。")
        if backend == "i2rt_can" and roles:
            raise ValueError("CAN 夹爪不允许通过 Web 绑定设备。")

        device_config = str(body.get("device_config") or "")
        self._repo_file(f"configs/{device_config}.yaml", "device_config")
        teleop_path = self._repo_file(body.get("teleop_config"), "teleop_config")
        saver_path = self._repo_file(body.get("saver_config"), "saver_config")
        canonical = json.dumps(
            body, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        revision = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        return SystemProfile(
            profile_id=profile_id,
            display_name=str(body.get("display_name") or profile_id),
            device_config=device_config,
            teleop_config=teleop_path,
            saver_config=str(saver_path.relative_to(REPO_ROOT)),
            require_recording_profile=bool(
                body.get("require_recording_profile", False)
            ),
            gripper_backend=backend,
            gripper_roles=roles,
            calibration_supported=bool(gripper.get("calibration_supported", False)),
            cameras=cameras,
            revision=revision,
        )

    def get(self, profile_id: str) -> SystemProfile:
        try:
            return self._profiles[profile_id]
        except KeyError as exc:
            raise ValueError(f"未知的 I2RT 系统方案：{profile_id}") from exc

    def find_by_device_config(self, device_config: str) -> SystemProfile | None:
        return next(
            (p for p in self._profiles.values() if p.device_config == device_config),
            None,
        )

    def public_profiles(self) -> list[dict[str, Any]]:
        return [profile.public_dict() for profile in self._profiles.values()]


__all__ = ["SystemProfile", "SystemProfileRegistry"]
