"""Machine-local hardware discovery, binding persistence, and launch snapshots."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from incubator.nedf3.I2RT_DM3.process_manager import REPO_ROOT
from incubator.nedf3.I2RT_DM3.system_profiles import SystemProfile

DEFAULT_BINDINGS_PATH = (
    REPO_ROOT / ".superinference" / "local" / "i2rt_hardware_bindings.json"
)
DEFAULT_LAUNCH_DIR = REPO_ROOT / ".superinference" / "local" / "i2rt_launches"
_SYNTHETIC_CAMERA_SHAPES = {"head": [480, 640, 3]}


@dataclass(frozen=True)
class HardwareCandidate:
    identity: str
    backend: str
    label: str
    device_key: str
    fingerprint: Mapping[str, str]

    def public_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "backend": self.backend,
            "label": self.label,
            "fingerprint": dict(self.fingerprint),
        }


@dataclass(frozen=True)
class HardwareInventory:
    candidates: Mapping[str, tuple[HardwareCandidate, ...]]
    warnings: tuple[str, ...] = ()

    def public_dict(self) -> dict[str, Any]:
        return {
            "candidates": {
                backend: [candidate.public_dict() for candidate in values]
                for backend, values in self.candidates.items()
            },
            "warnings": list(self.warnings),
        }

    def resolve(self, backend: str, identity: str) -> HardwareCandidate:
        for candidate in self.candidates.get(backend, ()):
            if candidate.identity == identity:
                return candidate
        raise ValueError(f"当前未发现 {backend} 设备：{identity}")


def _udev_properties(path: Path) -> dict[str, str]:
    try:
        output = subprocess.run(
            ["udevadm", "info", "--query=property", f"--name={path}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return {
        key: value
        for line in output.splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }


def _candidate_from_node(path: Path, backend: str) -> HardwareCandidate:
    properties = _udev_properties(path)
    fingerprint = {
        key: properties[key]
        for key in (
            "ID_VENDOR_ID",
            "ID_MODEL_ID",
            "ID_SERIAL",
            "ID_PATH",
            "ID_V4L_PRODUCT",
            "ID_V4L_CAPABILITIES",
        )
        if properties.get(key)
    }
    try:
        device_number = path.stat().st_rdev
        fallback_key = f"{os.major(device_number)}:{os.minor(device_number)}"
    except OSError:
        fallback_key = str(path)
    serial_number = properties.get("ID_SERIAL")
    usb_path = properties.get("ID_PATH")
    if serial_number and usb_path:
        device_key = f"{serial_number}|{usb_path}"
    else:
        device_key = serial_number or usb_path or fallback_key
    product = properties.get("ID_V4L_PRODUCT") or properties.get(
        "ID_MODEL_FROM_DATABASE"
    )
    label = f"{path} · {product}" if product else str(path)
    return HardwareCandidate(str(path), backend, label, device_key, fingerprint)


def _device_nodes(pattern: str) -> list[Path]:
    nodes: list[Path] = []
    for path in sorted(Path("/dev").glob(pattern)):
        try:
            if stat.S_ISCHR(path.stat().st_mode):
                nodes.append(path)
        except OSError:
            continue
    return nodes


def discover_hardware() -> HardwareInventory:
    """Enumerate only the hardware classes editable from the browser."""
    warnings: list[str] = []
    v4l2_candidates = tuple(
        _candidate_from_node(path, "v4l2") for path in _device_nodes("video[0-9]*")
    )
    v4l2 = tuple(
        candidate
        for candidate in v4l2_candidates
        if (
            candidate.fingerprint.get("ID_SERIAL")
            or candidate.fingerprint.get("ID_PATH")
        )
        and (
            not candidate.fingerprint.get("ID_V4L_CAPABILITIES")
            or ":capture:" in candidate.fingerprint["ID_V4L_CAPABILITIES"]
        )
    )
    serial_candidates = tuple(
        _candidate_from_node(path, "serial") for path in _device_nodes("ttyUSB[0-9]*")
    )
    serial = tuple(
        candidate
        for candidate in serial_candidates
        if candidate.fingerprint.get("ID_SERIAL")
        or candidate.fingerprint.get("ID_PATH")
    )
    if len(v4l2) < len(v4l2_candidates):
        warnings.append("已隐藏无稳定 udev 身份或非采集类型的视频节点。")
    if len(serial) < len(serial_candidates):
        warnings.append("已隐藏无稳定 udev 身份的 ttyUSB 节点。")
    realsense: list[HardwareCandidate] = []
    try:
        import pyrealsense2 as rs

        for device in rs.context().query_devices():
            serial_number = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            firmware = device.get_info(rs.camera_info.firmware_version)
            realsense.append(
                HardwareCandidate(
                    serial_number,
                    "realsense",
                    f"{name} · {serial_number}",
                    serial_number,
                    {"serial": serial_number, "name": name, "firmware": firmware},
                )
            )
    except Exception as exc:
        warnings.append(f"RealSense 枚举不可用：{exc}")
    return HardwareInventory(
        {"v4l2": v4l2, "serial": serial, "realsense": tuple(realsense)},
        tuple(warnings),
    )


class HardwareBindingStore:
    """Persist validated Web-editable bindings outside tracked configuration."""

    def __init__(self, path: Path = DEFAULT_BINDINGS_PATH) -> None:
        self.path = path

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "profiles": {}}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"无法读取本机硬件绑定：{exc}") from exc
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise RuntimeError("本机硬件绑定文件版本无效。")
        if not isinstance(value.get("profiles"), dict):
            raise RuntimeError("本机硬件绑定文件内容无效。")
        return value

    def get(self, profile_id: str) -> dict[str, Any] | None:
        binding = self._load()["profiles"].get(profile_id)
        if binding is None:
            return None
        if not isinstance(binding, dict) or set(binding) != {
            "profile_revision",
            "cameras",
            "grippers",
        }:
            raise RuntimeError("本机硬件绑定包含不支持的字段。")
        return dict(binding)

    @staticmethod
    def _selection(
        inventory: HardwareInventory,
        backend: str,
        identity: Any,
    ) -> dict[str, Any]:
        if not isinstance(identity, str) or not identity:
            raise ValueError("硬件路径或序列号不能为空。")
        candidate = inventory.resolve(backend, identity)
        return {
            "identity": candidate.identity,
            "device_key": candidate.device_key,
            "fingerprint": dict(candidate.fingerprint),
        }

    def save(
        self,
        profile: SystemProfile,
        *,
        cameras: Mapping[str, Any],
        grippers: Mapping[str, Any],
        inventory: HardwareInventory,
    ) -> dict[str, Any]:
        if set(cameras) != set(profile.cameras):
            raise ValueError("相机绑定必须完整包含当前系统方案定义的角色。")
        if set(grippers) != set(profile.gripper_roles):
            raise ValueError("夹爪绑定与当前系统方案不一致。")
        camera_values = {
            role: self._selection(inventory, camera.backend, cameras[role])
            for role, camera in profile.cameras.items()
        }
        gripper_values = {
            role: self._selection(inventory, "serial", grippers[role])
            for role in profile.gripper_roles
        }
        keys = [
            entry["device_key"]
            for entry in (*camera_values.values(), *gripper_values.values())
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("同一个物理设备不能绑定到多个角色。")
        binding = {
            "profile_revision": profile.revision,
            "cameras": camera_values,
            "grippers": gripper_values,
        }
        value = self._load()
        value["profiles"][profile.profile_id] = binding
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.parent.chmod(0o700)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            temporary.chmod(0o600)
            temporary.replace(self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink(missing_ok=True)
        return binding

    def validate(
        self,
        profile: SystemProfile,
        inventory: HardwareInventory,
    ) -> dict[str, Any]:
        binding = self.get(profile.profile_id)
        if binding is None:
            raise RuntimeError("请先保存当前系统方案的本机硬件绑定。")
        if binding.get("profile_revision") != profile.revision:
            raise RuntimeError("系统方案已更新，请重新确认并保存本机硬件绑定。")
        expected_roles = {"cameras": profile.cameras, "grippers": profile.gripper_roles}
        for group, roles in expected_roles.items():
            values = binding.get(group)
            if not isinstance(values, dict) or set(values) != set(roles):
                raise RuntimeError("本机硬件绑定不完整，请重新保存。")
            for role, saved in values.items():
                backend = (
                    profile.cameras[role].backend if group == "cameras" else "serial"
                )
                if not isinstance(saved, dict):
                    raise RuntimeError("本机硬件绑定内容无效。")
                current = inventory.resolve(backend, str(saved.get("identity") or ""))
                if current.device_key != saved.get("device_key") or dict(
                    current.fingerprint
                ) != saved.get("fingerprint"):
                    raise RuntimeError(f"{role} 的设备身份已变化，请重新确认硬件绑定。")
        return binding


@dataclass(frozen=True)
class LaunchPlan:
    profile: SystemProfile
    teleop_config: Path
    saver_config: Path
    environment: Mapping[str, str]


def _write_yaml_snapshot(value: Mapping[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            yaml.safe_dump(value, handle, allow_unicode=True, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.chmod(0o600)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)
    return destination


def _profile_saver_config(
    profile: SystemProfile,
    launch_dir: Path,
) -> Path:
    source = REPO_ROOT / profile.saver_config
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid saver config: {profile.saver_config}")
    devices = value.get("devices")
    cameras = devices.get("cameras") if isinstance(devices, dict) else None
    if not isinstance(cameras, dict):
        raise ValueError(
            f"saver config has no devices.cameras mapping: {profile.saver_config}"
        )
    if "head" in profile.cameras:
        if not cameras.get("head"):
            raise ValueError(
                f"three-camera profile has no head SHM: {profile.saver_config}"
            )
        return source

    synthetic_shapes = value.get("synthetic_camera_shapes")
    if synthetic_shapes is None:
        synthetic_shapes = {}
        value["synthetic_camera_shapes"] = synthetic_shapes
    if not isinstance(synthetic_shapes, dict):
        raise ValueError(
            f"invalid synthetic_camera_shapes mapping: {profile.saver_config}"
        )
    already_synthetic = (
        cameras.get("head") is None
        and synthetic_shapes.get("head") == _SYNTHETIC_CAMERA_SHAPES["head"]
    )
    if already_synthetic:
        return source
    cameras["head"] = None
    synthetic_shapes["head"] = _SYNTHETIC_CAMERA_SHAPES["head"]
    return _write_yaml_snapshot(
        value,
        launch_dir / f"{profile.profile_id}.saver.yaml",
    )


def build_launch_plan(
    profile: SystemProfile,
    binding: Mapping[str, Any],
    launch_dir: Path = DEFAULT_LAUNCH_DIR,
) -> LaunchPlan:
    environment = {
        camera.environment: str(binding["cameras"][role]["identity"])
        for role, camera in profile.cameras.items()
    }
    if any(camera.backend == "realsense" for camera in profile.cameras.values()):
        environment["SI_I2RT_HEAD_CAMERA_ENABLED"] = str(
            "head" in profile.cameras
        ).lower()
    teleop_path = profile.teleop_config
    if profile.gripper_backend == "lktech":
        value = yaml.safe_load(profile.teleop_config.read_text(encoding="utf-8"))
        for role in profile.gripper_roles:
            value["arms"][role]["follower"]["gripper"]["port"] = binding["grippers"][
                role
            ]["identity"]
        teleop_path = launch_dir / f"{profile.profile_id}.teleop.yaml"
        _write_yaml_snapshot(value, teleop_path)
    else:
        environment["SI_I2RT_YAM_GRIPPER_TYPE"] = "crank_4310"
    environment["SI_I2RT_TELEOP_CONFIG_PATH"] = str(teleop_path)
    saver_path = _profile_saver_config(profile, launch_dir)
    return LaunchPlan(profile, teleop_path, saver_path, environment)


__all__ = [
    "HardwareBindingStore",
    "HardwareCandidate",
    "HardwareInventory",
    "LaunchPlan",
    "build_launch_plan",
    "discover_hardware",
]
