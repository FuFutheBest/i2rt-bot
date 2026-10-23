#!/usr/bin/env python3
"""Interactive hardware-to-config binding used before a rollout.

The operating system assigns ``ttyUSB`` and ``video`` numbers at boot.  This
workflow lets an operator identify each physical hand/camera and persists that
mapping in the supplied YAML device config.  It deliberately runs in a
predictable order: serial devices first, then cameras.

Usage::

    python scripts/diagnostics/rollout_device_preflight.py CONFIG.yaml
    python scripts/diagnostics/rollout_device_preflight.py CONFIG.yaml \
        --gripper_calibrate
"""

from __future__ import annotations

import argparse
import glob
import ipaddress
import json
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from incubator.control_panel.process_cleanup import (  # noqa: E402
    ProcessCleanupError,
    cleanup_before_startup,
    force_cleanup_requested,
)

SIDE_NAMES = ("left", "right")
SIDE_LABELS = {"left": "左", "right": "右"}
SIDE_KEYS = {"l": "left", "r": "right", "left": "left", "right": "right"}
_ROBOT_SELECTION_LABEL_ALPHABET = "abcdefghijklmnopqrtuvwxyz"


class RolloutPreflightError(ValueError):
    """Raised when a config or operator binding is invalid."""


@dataclass(frozen=True)
class GripperSpec:
    port: str
    motor_id: int
    calibration_file: str | None
    max_width: float
    gripper_type: str
    settings: Mapping[str, Any] = ()


def discover_serial_ports(dev_root: str | Path = "/dev") -> list[str]:
    """Return present ttyUSB character-device names in stable numeric order."""
    root = Path(dev_root)
    paths = [Path(p) for p in glob.glob(str(root / "ttyUSB*"))]

    def sort_key(path: Path) -> tuple[int, str]:
        suffix = path.name.removeprefix("ttyUSB")
        return (int(suffix), path.name) if suffix.isdigit() else (10**9, path.name)

    return [str(path) for path in sorted(paths, key=sort_key) if path.is_char_device()]


def discover_video_devices(dev_root: str | Path = "/dev") -> list[str]:
    """Return present /dev/videoN nodes in numeric order."""
    root = Path(dev_root)
    paths = [Path(p) for p in glob.glob(str(root / "video*"))]

    def sort_key(path: Path) -> tuple[int, str]:
        suffix = path.name.removeprefix("video")
        return (int(suffix), path.name) if suffix.isdigit() else (10**9, path.name)

    return [str(path) for path in sorted(paths, key=sort_key) if path.is_char_device()]


def _pair(value: Any, default: Any = None) -> list[Any]:
    if isinstance(value, (list, tuple)):
        values = list(value)
    elif value is None:
        values = []
    else:
        values = [value]
    return (values + [default, default])[:2]


def _device_config_path(doc: Mapping[str, Any], top_level_path: Path) -> Path:
    camera = doc.get("camera")
    if not isinstance(camera, Mapping) or not {"left_capture_id", "right_capture_id"} <= set(camera):
        raise RolloutPreflightError(
            "请使用包含 camera.left_capture_id/right_capture_id 的顶层 rollout 配置；"
            "不要传入 configs/devices/ 下的设备 YAML"
        )
    defaults = doc.get("defaults")
    if not isinstance(defaults, list):
        raise RolloutPreflightError("顶层 rollout 配置缺少 defaults 列表")
    for entry in defaults:
        if not isinstance(entry, Mapping):
            continue
        for group, name in entry.items():
            group_name = str(group).lstrip("/").split("@", maxsplit=1)[0]
            if group_name in {"devices", "devices/infer"} and isinstance(name, str):
                path = REPO_ROOT / "configs" / group_name / f"{name}.yaml"
                if not path.is_file():
                    raise RolloutPreflightError(f"找不到引用的设备配置： {path}")
                return path
    raise RolloutPreflightError("顶层 rollout 配置未引用设备配置")


def _gripper_block(doc: Mapping[str, Any]) -> dict[str, Any]:
    devices = doc.get("devices", [])
    if not isinstance(devices, list):
        raise RolloutPreflightError("config.devices 必须是列表")
    for device in devices:
        if isinstance(device, dict) and isinstance(device.get("gripper"), dict):
            return device["gripper"]
    raise RolloutPreflightError("配置中没有包含 gripper 设置的设备")


def _robot_block(doc: Mapping[str, Any]) -> dict[str, Any]:
    devices = doc.get("devices", [])
    if not isinstance(devices, list):
        raise RolloutPreflightError("config.devices 必须是列表")
    for device in devices:
        if isinstance(device, dict) and device.get("class") == "RizonRobot":
            return device
    raise RolloutPreflightError("配置中没有 RizonRobot 设备")


def discover_robot_hardware() -> Any:
    """Read Flexiv Elements cache and local interfaces without opening RDK."""
    from incubator.control_panel.hardware_discovery import HardwareDiscovery

    return HardwareDiscovery().discover()


def _unique_robot_for_model(robots: Sequence[Mapping[str, Any]], model: str) -> Mapping[str, Any] | None:
    candidates = [robot for robot in robots if robot.get("model") == model]
    active = [robot for robot in candidates if robot.get("peer_active") is True]
    pool = active or candidates
    return pool[0] if len(pool) == 1 else None


def _robot_selection_label(index: int) -> str:
    """Create a stable candidate label while reserving ``s`` for skip."""
    number = index + 1
    label = ""
    while number:
        number, remainder = divmod(number - 1, len(_ROBOT_SELECTION_LABEL_ALPHABET))
        label = _ROBOT_SELECTION_LABEL_ALPHABET[remainder] + label
    return label


def update_robot_binding_from_elements(
    device_doc: Mapping[str, Any],
    snapshot: Any,
    *,
    input_fn: Callable[[str], str] | None = None,
) -> list[str]:
    """Update robot serials and the unique robot-facing local IPv4 address."""
    robot_block = _robot_block(device_doc)
    by_serial: dict[str, Mapping[str, Any]] = {}
    for item in getattr(snapshot, "robots", []):
        if isinstance(item, Mapping):
            serial = str(item.get("rdk_sn") or item.get("serial_number") or "").strip()
            if serial and (serial not in by_serial or item.get("peer_active") is True):
                by_serial[serial] = item
    robots = list(by_serial.values())
    configured = robot_block.get("robot_sn")
    old_serials = [configured] if isinstance(configured, str) else list(configured or [])
    count = len(old_serials)
    selected = None
    if input_fn is not None and len(robots) > 1 and count:
        options = {}
        print("Elements 缓存中的机器人候选（缓存记录不代表当前已连接）：")
        for index, robot in enumerate(robots):
            label = _robot_selection_label(index)
            options[label] = robot
            print(
                f"  {label}) {robot.get('rdk_sn') or robot.get('serial_number')} "
                f"型号={robot.get('model', '?')} IP={robot.get('ip', '?')}"
            )
        order = "左臂、右臂" if count == 2 else "robot_sn 配置排列"
        while True:
            try:
                answer = input_fn(
                    f"请按{order}顺序选择 {count} 个机器人，用逗号或空格分隔"
                    f"（当前配置：{old_serials}；输入 s/skip 保留当前配置，abort 退出）："
                ).strip().casefold()
            except EOFError as exc:
                raise RolloutPreflightError("机器人选择尚未确认，输入已结束") from exc
            if answer == "abort":
                raise RolloutPreflightError("操作员已终止机器人绑定")
            if answer in {"s", "skip"}:
                return [
                    "已跳过机器人绑定，保留当前 robot_sn 和 "
                    "network_interface_whitelist 配置。"
                ]
            if len(robots) < count:
                raise RolloutPreflightError(f"需要 {count} 个机器人序列号，但只发现 {len(robots)} 个候选")
            keys = answer.replace(",", " ").replace("，", " ").split()
            if len(keys) != count or len(set(keys)) != count or any(key not in options for key in keys):
                print(f"请按所需顺序选择恰好 {count} 个不同的有效选项。")
                continue
            selected = [options[key] for key in keys]
            break
    left = _unique_robot_for_model(robots, "Rizon4")
    right = _unique_robot_for_model(robots, "Rizon4R")
    messages: list[str] = []
    if selected is None and count == 1 and len(robots) == 1:
        selected = robots
    if selected is None and (count != 2 or left is None or right is None):
        inventory = ", ".join(
            str(item.get("rdk_sn") or item.get("serial_number")) for item in robots
        ) or "无"
        messages.append(
            "机器人配置未修改：Elements 缓存无法唯一确定一个 "
            f"Rizon4 和一个 Rizon4R 候选（发现：{inventory}）。"
        )
        return messages

    if selected is None:
        selected = [left, right]
    serials = [str(robot.get("rdk_sn") or robot.get("serial_number")) for robot in selected]
    if not all(serials):
        messages.append("机器人配置未修改：选中的 Elements 记录没有序列号。")
        return messages
    robot_block["robot_sn"] = serials[0] if isinstance(configured, str) else serials
    if old_serials != serials:
        messages.append(f"已更新设备配置的 robot_sn： {old_serials!r} -> {serials!r}")
    else:
        messages.append(f"已确认 Flexiv Elements 缓存中的 robot_sn： {serials!r}")

    robot_ips = []
    for robot in selected:
        try:
            robot_ips.append(ipaddress.ip_address(str(robot.get("ip") or "")))
        except ValueError:
            robot_ips = []
            break
    matching_interfaces: list[Mapping[str, Any]] = []
    if robot_ips:
        for interface in getattr(snapshot, "interfaces", []):
            if not isinstance(interface, Mapping):
                continue
            try:
                network = ipaddress.ip_network(str(interface.get("network")), strict=False)
                local_ip = ipaddress.ip_address(str(interface.get("ipv4")))
            except ValueError:
                continue
            if local_ip.version == robot_ips[0].version and all(ip in network for ip in robot_ips):
                matching_interfaces.append(interface)
    if len(matching_interfaces) == 1:
        interface = matching_interfaces[0]
        whitelist = [str(interface["ipv4"])]
        old_whitelist = list(robot_block.get("network_interface_whitelist") or [])
        robot_block["network_interface_whitelist"] = whitelist
        if old_whitelist != whitelist:
            messages.append(
                "已更新设备配置的 network_interface_whitelist："
                f"{old_whitelist!r} -> {whitelist!r} "
                f"({interface.get('name')}, {interface.get('network')})"
            )
        else:
            messages.append(
                "已确认机器人通信网卡："
                f"{interface.get('name')} {whitelist[0]} ({interface.get('network')})"
            )
    else:
        messages.append(
            "network_interface_whitelist 未修改：无法唯一确定一个"
            "网段覆盖所有所选机器人 IP 的本机网卡。"
        )
    return messages


def _resolve_calibration_path(raw: str, config_path: Path) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _specs_from_config(
    doc: Mapping[str, Any],
    repository_root: Path,
    config_path: Path | None = None,
) -> list[GripperSpec]:
    block = _gripper_block(doc)
    ports = _pair(block.get("port"))
    motor_ids = _pair(block.get("motor_id"), 1)
    files = _pair(block.get("calibration_files"))
    types = str(block.get("gripper_type", "soft"))
    specs: list[GripperSpec] = []
    for port, motor_id, calibration_file in zip(ports, motor_ids, files):
        try:
            motor = int(motor_id or 1)
        except (TypeError, ValueError):
            motor = 1
        configured_max_width = block.get("max_width")
        max_width = 0.0925
        if isinstance(configured_max_width, list):
            index = len(specs)
            values = configured_max_width
            if index < len(values) and values[index] is not None:
                max_width = float(values[index])
        elif configured_max_width is not None:
            max_width = float(configured_max_width)
        if calibration_file:
            path = Path(str(calibration_file))
            if not path.is_absolute():
                path = _resolve_calibration_path(str(calibration_file), config_path or repository_root)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                calibration_data = data if isinstance(data, dict) else {}
                if calibration_data.get("max_width") is not None:
                    max_width = float(calibration_data["max_width"])
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                calibration_data = {}
        else:
            calibration_data = {}
        if not calibration_data:
            try:
                from utils.lk_gripper_presets import get_lk_gripper_preset

                calibration_data = get_lk_gripper_preset(types)
            except (ImportError, ValueError):
                calibration_data = {}
        specs.append(
            GripperSpec(
                str(port or ""),
                motor,
                str(calibration_file) if calibration_file else None,
                max_width,
                types,
                calibration_data,
            )
        )
    return specs


def _calibration_args(gripper_type: str) -> Any:
    from argparse import Namespace

    return Namespace(
        gripper_type=gripper_type,
        min_torque_iq=None,
        max_torque_iq=None,
        arm_length_m=None,
        apex_angle_max_deg=None,
        max_width=None,
        gear_ratio=None,
        baudrate=115200,
        speed_dps=180.0,
        read_timeout=0.1,
        wait_timeout=15.0,
        calibration_timeout=15.0,
        calibration_settle_time=0.24,
        calibration_cycles=1,
        calibration_dir=None,
        name="rollout",
    )


def _identification_spec(port: str, specs: Sequence[GripperSpec]) -> GripperSpec:
    """Match calibration to its recorded port, never to USB enumeration order."""
    matches = []
    for spec in specs:
        settings = spec.settings if isinstance(spec.settings, Mapping) else {}
        if str(settings.get("port") or spec.port) == port:
            matches.append(spec)
    if len(matches) != 1:
        raise RolloutPreflightError(
            f"无法为 {port} 唯一匹配标定；请使用 --gripper_calibrate "
            "重新标定并确认当前串口绑定"
        )
    return matches[0]


def calibrate_gripper(port: str, spec: GripperSpec, side: str) -> tuple[float, float, float]:
    """Run the repository's real LK calibration program for one port."""
    from scripts.calibration import calibrate_lk_gripper

    result = calibrate_lk_gripper.calibrate_gripper(
        _calibration_args(spec.gripper_type), port, spec.motor_id, side
    )
    if result is None:
        raise RolloutPreflightError(f"{port} 标定失败")
    return result


def exercise_gripper(
    port: str,
    spec: GripperSpec,
    *,
    settle_s: float = 0.5,
    stop_event: threading.Event | None = None,
) -> None:
    """Repeat calibrated 30%-70% opening targets until identification stops."""
    from third_party.lktech_motor_sdk.example.lktech_gripper import Gripper

    settings = dict(spec.settings) if isinstance(spec.settings, Mapping) else {}
    kwargs = {
        "min_torque_iq": settings.get("min_torque_iq", -40),
        "max_torque_iq": settings.get("max_torque_iq", 190),
        "arm_length_m": settings.get("arm_length_m", 0.06543),
        "apex_angle_max_deg": settings.get("apex_angle_max_deg", 90.0),
        "gear_ratio": settings.get("gear_ratio", 10.0),
        "min_angle_deg": settings.get("min_angle_deg"),
        "max_angle_deg": settings.get("max_angle_deg"),
    }
    gripper = Gripper(
        robot=None,
        port=port,
        motor_id=spec.motor_id,
        baudrate=115200,
        auto_calibrate=False,
        **kwargs,
    )
    try:
        gripper.Enable(f"rollout-{port}")
        velocity = 0.3
        while True:
            for ratio in (0.3, 0.7):
                if stop_event is not None and stop_event.is_set():
                    return
                gripper.Move(max(0.0, min(1.0, ratio)) * spec.max_width, velocity)
                if stop_event is None:
                    time.sleep(settle_s)
                elif stop_event.wait(settle_s):
                    return
            if stop_event is None:
                return
    finally:
        try:
            gripper.Disable()
        except Exception:
            pass


def _ask_side(prompt: str, input_fn: Callable[[str], str]) -> str:
    answer = input_fn(prompt).strip().casefold()
    if answer not in SIDE_KEYS:
        raise RolloutPreflightError("请输入 l（左）或 r（右）")
    return SIDE_KEYS[answer]


def identify_gripper(port: str, spec: GripperSpec, input_fn: Callable[[str], str]) -> str:
    """Keep a gripper moving until the operator identifies its side."""
    stop_event = threading.Event()
    failures: list[BaseException] = []

    def move_until_answered() -> None:
        try:
            exercise_gripper(port, spec, stop_event=stop_event)
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=move_until_answered, name=f"identify-{Path(port).name}", daemon=True)
    worker.start()
    try:
        side = _ask_side(f"{port} 上正在运动的是哪只夹爪？[l/r]（l=左，r=右）：", input_fn)
    finally:
        stop_event.set()
        worker.join(timeout=3.0)
    if worker.is_alive():
        raise RolloutPreflightError(f"等待 {port} 停止识别动作超时")
    if failures:
        raise RolloutPreflightError(f"{port} 夹爪识别动作失败： {failures[0]}")
    return side


def identify_camera(video: str) -> str | None:
    """Preview a camera; return its l/r side, None to skip, or raise to abort."""
    try:
        import cv2
    except ImportError as exc:
        raise RolloutPreflightError("相机预览需要安装 opencv-python") from exc

    capture = cv2.VideoCapture(video, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        print(f"跳过 {video}：该视频节点无法采集图像。")
        return None
    window = video
    print(f"正在预览 {video}：请点击图像窗口后按 L=左相机、R=右相机、S=跳过、Q/Esc=退出。")
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print(f"跳过 {video}：设备已打开，但未获取到图像。")
                return None
            cv2.putText(
                frame,
                f"{video}  [L] [R] [S] [Q/Esc]",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, frame)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("l"), ord("L")):
                return "left"
            if key in (ord("r"), ord("R")):
                return "right"
            if key in (ord("s"), ord("S")):
                print(f"跳过 {video}：操作员选择跳过。")
                return None
            if key in (ord("q"), ord("Q"), 27):
                raise RolloutPreflightError("操作员已终止相机绑定")
    except cv2.error as exc:
        raise RolloutPreflightError(f"无法显示 {video} 的相机预览： {exc}") from exc
    finally:
        capture.release()
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


def _set_pair(mapping: dict[str, Any], key: str, index: int, value: Any) -> None:
    values = _pair(mapping.get(key))
    values[index] = value
    mapping[key] = values


def _write_calibration_json(path: Path, side: str, port: str, motor_id: int, result: Sequence[float]) -> None:
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError):
            pass
    data.update({"label": data.get("label") or side, "port": port, "motor_id": motor_id,
                 "min_angle_deg": float(result[0]), "max_angle_deg": float(result[1]),
                 "gear_ratio": float(result[2])})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def bind_devices(
    config_path: str | Path,
    *,
    gripper_calibrate: bool = False,
    serial_ports: Sequence[str] | None = None,
    video_devices: Sequence[str] | None = None,
    input_fn: Callable[[str], str] = input,
    camera_side_fn: Callable[[str], str | None] = identify_camera,
    robot_discovery_fn: Callable[[], Any] = discover_robot_hardware,
) -> Path:
    """Run the interactive binding and persist the updated YAML path."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise RolloutPreflightError(f"找不到配置文件： {path}")
    try:
        from ruamel.yaml import YAML
    except ImportError as exc:
        raise RolloutPreflightError("保存配置并保留注释需要安装 ruamel.yaml") from exc
    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True
    yaml_rt.indent(mapping=2, sequence=4, offset=2)
    yaml_rt.width = 4096
    yaml_rt.representer.add_representer(
        type(None), lambda representer, _: representer.represent_scalar("tag:yaml.org,2002:null", "null")
    )
    with path.open("r", encoding="utf-8") as stream:
        top_level_doc = yaml_rt.load(stream)
    if not isinstance(top_level_doc, dict):
        raise RolloutPreflightError("配置根节点必须是映射")
    device_path = _device_config_path(top_level_doc, path)
    with device_path.open("r", encoding="utf-8") as stream:
        device_doc = yaml_rt.load(stream)
    if not isinstance(device_doc, dict):
        raise RolloutPreflightError("设备配置根节点必须是映射")

    from utils.device_identity import validate_device_class_ids

    try:
        validate_device_class_ids(device_doc.get("devices", []))
    except ValueError as exc:
        raise RolloutPreflightError(str(exc)) from exc
    logical_identity = [
        (device["class"], device["device_class_id"])
        for device in device_doc["devices"]
    ]

    try:
        robot_snapshot = robot_discovery_fn()
    except (OSError, RuntimeError, RolloutPreflightError) as exc:
        robot_messages = [f"机器人配置未修改：读取 Elements 缓存失败： {exc}"]
    else:
        robot_messages = update_robot_binding_from_elements(device_doc, robot_snapshot, input_fn=input_fn)

    ports = list(serial_ports if serial_ports is not None else discover_serial_ports())
    if len(ports) < 2:
        raise RolloutPreflightError(f"需要两只夹爪，但只发现 {len(ports)} 个 ttyUSB 设备")
    specs = _specs_from_config(device_doc, REPO_ROOT, device_path)
    identification_specs = (
        {} if gripper_calibrate else {port: _identification_spec(port, specs) for port in ports[:2]}
    )
    block = _gripper_block(device_doc)
    assigned: dict[str, str] = {}
    calibration_results: dict[str, tuple[float, float, float]] = {}
    for index, port in enumerate(ports[:2]):
        spec = specs[index] if index < len(specs) else GripperSpec(port, 1, None, 0.0925, "soft")
        if gripper_calibrate:
            result = calibrate_gripper(port, spec, f"rollout-{index + 1}")
            print(f"{port} 标定完成。")
            side = _ask_side(f"{port} 上刚才运动的是哪只夹爪？[l/r]（l=左，r=右）：", input_fn)
            if side in assigned:
                raise RolloutPreflightError(f"两只夹爪被标记为同一侧：{SIDE_LABELS[side]}")
            assigned[side] = port
            calibration_results[side] = result
        else:
            side = identify_gripper(port, identification_specs[port], input_fn)
            if side in assigned:
                raise RolloutPreflightError(f"两只夹爪被标记为同一侧：{SIDE_LABELS[side]}")
            assigned[side] = port
    for index, side in enumerate(SIDE_NAMES):
        _set_pair(block, "port", index, assigned[side])
        if gripper_calibrate and index < len(specs) and specs[index].calibration_file:
            calibration_path = Path(specs[index].calibration_file)
            if not calibration_path.is_absolute():
                calibration_path = _resolve_calibration_path(str(calibration_path), device_path)
            _write_calibration_json(
                calibration_path,
                side,
                assigned[side],
                specs[index].motor_id,
                calibration_results[side],
            )

    videos = list(video_devices if video_devices is not None else discover_video_devices())
    assigned_cameras: dict[str, int] = {}
    for video in videos:
        side = camera_side_fn(video)
        if side is None:
            continue
        if side not in SIDE_NAMES:
            raise RolloutPreflightError(f"相机预览返回了无效的左右标识：{side!r}")
        if side in assigned_cameras:
            raise RolloutPreflightError(f"两个相机被标记为同一侧：{SIDE_LABELS[side]}")
        suffix = Path(video).name.removeprefix("video")
        if not suffix.isdigit():
            raise RolloutPreflightError(f"无法从 {video} 提取相机编号")
        assigned_cameras[side] = int(suffix)
        if len(assigned_cameras) == 2:
            break
    if set(assigned_cameras) != set(SIDE_NAMES):
        raise RolloutPreflightError(
            f"需要选齐左右两路相机，目前已识别：{[SIDE_LABELS[side] for side in sorted(assigned_cameras)]}"
        )
    camera = top_level_doc["camera"]
    camera["left_capture_id"] = assigned_cameras["left"]
    camera["right_capture_id"] = assigned_cameras["right"]

    if logical_identity != [
        (device["class"], device["device_class_id"])
        for device in device_doc["devices"]
    ]:
        raise RolloutPreflightError("预检不能修改 device_class_id 或逻辑设备顺序")
    validate_device_class_ids(device_doc["devices"])

    with device_path.open("w", encoding="utf-8") as stream:
        yaml_rt.dump(device_doc, stream)
    with path.open("w", encoding="utf-8") as stream:
        yaml_rt.dump(top_level_doc, stream)
    for message in robot_messages:
        print(f"设备预检：{message}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="识别机器人、USB 夹爪和相机，并将绑定写入设备配置。", add_help=False
    )
    parser._positionals.title = "位置参数"
    parser._optionals.title = "可选参数"
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    parser.add_argument("config", help="顶层 Hydra rollout 配置文件路径")
    parser.add_argument(
        "gripper_calibrate_value",
        nargs="?",
        choices=("true", "false", "1", "0", "yes", "no"),
        help="可选的夹爪标定开关",
    )
    parser.add_argument(
        "--gripper_calibrate",
        "--gripper-calibrate",
        nargs="?",
        const="true",
        choices=("true", "false", "1", "0", "yes", "no"),
        default=None,
        help="先标定 LK 夹爪，再确认左右",
    )
    parser.add_argument(
        "--force-process-cleanup",
        action="store_true",
        help="正常清理失败后，强制终止相关启动器进程。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_toggle = args.gripper_calibrate if args.gripper_calibrate is not None else args.gripper_calibrate_value
    calibrate = str(raw_toggle).casefold() in {"true", "1", "yes"} if raw_toggle is not None else False
    try:
        force_cleanup = args.force_process_cleanup or force_cleanup_requested()
        if force_cleanup:
            cleanup_report = cleanup_before_startup("rollout_preflight", force=True)
        else:
            cleanup_report = cleanup_before_startup("rollout_preflight")
        if cleanup_report.candidates:
            print(
                f"设备预检：进程清理：发现 {len(cleanup_report.candidates)} 个相关进程，"
                f"已停止 {len(cleanup_report.terminated_pids)} 个，"
                f"已停止的 PID：{cleanup_report.terminated_pids}"
            )
        bind_devices(args.config, gripper_calibrate=calibrate)
    except ProcessCleanupError as exc:
        print("设备预检：错误：相关进程清理未完成。", file=sys.stderr)
        for process in exc.report.survivors:
            print(f"仍在运行：PID {process.pid}，命令：{process.command}", file=sys.stderr)
        for pid, error in exc.report.errors.items():
            print(f"清理失败：PID {pid}，原因：{error}", file=sys.stderr)
        return 2
    except (RolloutPreflightError, OSError, RuntimeError) as exc:
        print(f"设备预检：错误： {exc}", file=sys.stderr)
        return 2
    print(f"已更新 rollout 配置： {Path(args.config).expanduser().resolve()}")
    print("可以直接启动推理，无须修改配置")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
