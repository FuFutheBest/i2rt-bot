#!/usr/bin/env python3
"""R1 NEDF3 data-collection saver (new HAL arm/hand/head schema).

Records the topic layout described in
``docs/R1遥操作系统数据样例（NEDF3）.md`` — distinct from the legacy UMI saver
(``async_data_saver_NEDF3.py``), which is left untouched so other users keep
their pipeline.

Per side (right/left Flexiv arm):
    nmx/hal/camera/camera_{0,1}/rgbd
    nmx/hal/arm/{side}/tcp/cmd     <- COMMANDED Cartesian target (ArmTcpCmd)
    nmx/hal/arm/{side}/tcp/state   <- MEASURED tcp pose          (ArmTcpState)
    nmx/hal/arm/{side}/tcp/vel     <- MEASURED tcp velocity 6D   (ArmTcpState linear/angular_vel)
    nmx/hal/arm/{side}/joint/state <- MEASURED joint position q  (ArmState joints[i].pos)
    nmx/hal/arm/{side}/joint/vel   <- MEASURED joint velocity dq (ArmState joints[i].vel)
    nmx/hal/arm/{side}/eef_wrench  <- ext wrench [Fx,Fy,Fz,Wx,Wy,Wz] (ArmEndForce)
    nmx/hal/arm/{side}/extra       <- empty Metadata
    nmx/hal/hand/{side}/cmd        <- gripper COMMANDED width_m   (HandCmd)
    nmx/hal/hand/{side}/state      <- gripper MEASURED width_m    (HandState)
    nmx/hal/hand/{side}/extra      <- empty Metadata

Head (YAM arm):
    nmx/hal/camera/camera_2/rgbd
    nmx/hal/head/cmd               <- COMMANDED FK pose          (ArmTcpCmd)
    nmx/hal/head/state             <- MEASURED FK pose           (ArmTcpState)
    nmx/hal/head/joint/state       <- measured joint angles      (ArmState)
    nmx/hal/head/extra             <- empty Metadata

Requires the wider R1 device layout published by devices/robot.py:
    RizonRobotLKForceGated data_shape [88]; YamHeadRobot data_shape [20]
(see configs/devices/quest3_collect_r1.yaml). With a narrower frame the saver
falls back gracefully (hand/cmd == hand/state, zero wrench).

NOTE: The backward-compatible base block carries MEASURED TCP poses. The R1
extension carries COMMANDED TCP poses, wrench, velocity, and gripper command
widths. When an extension is absent, command topics fall back to measured state.

运行方法 (Usage)
================
前提:先在另一个终端用 device_starter 把设备拉起来(它会创建 device_summary_data
共享内存,本脚本从中读数据)::

    python3 scripts/diagnostics/runtime_profiles.py run \
        --config-name quest3_collect_r1 --role device -- \
        device_starter.py --config-path configs --config-name quest3_collect_r1

设备起来后,再开一个终端跑本采集脚本:

1) 手柄按钮控制 episode(默认,推荐):需要 --config 提供 runtime.signal_files,
   用 Quest 手柄按钮开始/结束 episode::

       python3 scripts/diagnostics/runtime_profiles.py run \
           --config-name quest3_collect_r1 --role recording -- \
           -m incubator.r1.async_data_saver_NEDF3_r1 \
           --config configs/quest3_collect_r1.yaml

2) 键盘(回车)控制 episode:保留 --config 以记录真实运行元数据,再加
   --disable-button-control::

       python3 scripts/diagnostics/runtime_profiles.py run \
           --config-name quest3_collect_r1 --role recording -- \
           -m incubator.r1.async_data_saver_NEDF3_r1 \
           --config configs/quest3_collect_r1.yaml \
           --disable-button-control

常用参数:
    --config <yaml>          Hydra 配置(用于信号文件和真实机器人元数据)
    -o, --output-dir <dir>   保存目录(默认 saved_data),每条 episode 存为 <uuid>/
    -t, --task-name <name>   任务名(写入 NEDF metadata,默认 data_collection)
    -r, --robot-type <s>     robot_vendor_model(默认 R1-Flexiv-Rizon4)
    --disable-button-control 改用键盘回车开始/结束 episode
    --disable-anomaly-detection  关闭录后异常检测
    -s, --summary-shm <name> 设备汇总共享内存名(默认 device_summary_data)
    --status                 打印状态后退出

录完的数据(<output-dir>/<uuid>/)可用 r1_teleop_dm3_upload 上传云端::

    python3 scripts/diagnostics/runtime_profiles.py run \
        --config-name quest3_collect_r1 --role upload -- \
        -m incubator.r1.r1_teleop_dm3_upload.cli upload saved_data
"""

import argparse
import os
import sys
import time
import uuid
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

# NEDF SDK
from dm3_core.nedf import (
    NEDFFactory,
    NEDFWriterConfig,
    NEDFMetadata,
    NEDFTopicsConfig,
    NEDFType,
)

from utils.logger_config import logger
import json
import yaml
try:
    from pynput import keyboard as pynput_keyboard
except (ImportError, OSError, RuntimeError):
    pynput_keyboard = None
from nmx_msg.Image_pb2 import Image, ImageDataType, RGBD
from utils.time_control import precise_loop_timing
from incubator.control_panel.episode_anomaly_detector import EpisodeAnomalyDetector
from incubator.r1.rizon_state_layout import RizonDualArmStateLayout
from incubator.r1.async_write_lifecycle import AsyncWriteLifecycleMixin

_GIT_COMMIT_ID_CACHE: Optional[str] = None
_INCOMPLETE_EPISODE_MARKER = "INCOMPLETE.json"


def get_git_commit_id() -> str:
    """Return the current repo HEAD commit id (full hash), or "" if unavailable.

    Resolved once from the repository root so it works regardless of cwd, and
    cached for the process lifetime since the codebase version is fixed per run.
    """
    global _GIT_COMMIT_ID_CACHE
    if _GIT_COMMIT_ID_CACHE is not None:
        return _GIT_COMMIT_ID_CACHE
    try:
        commit_id = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as exc:
        logger.warning(f"Failed to resolve git commit id for codebase_version: {exc}")
        commit_id = ""
    _GIT_COMMIT_ID_CACHE = commit_id
    return commit_id


def load_composed_config(
    config_path: str,
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compose a repository Hydra config and return a resolved plain dictionary."""
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf

    config_name_with_ext = os.path.basename(config_path)
    config_name = os.path.splitext(config_name_with_ext)[0]
    configs_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=configs_dir, version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=list(overrides or []))
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError(f"Hydra config {config_name!r} did not resolve to a dictionary")
    return resolved


def load_signal_files_from_config(config_path: str) -> Dict[str, str]:
    """Load signal file patterns from a composed Hydra config."""
    config = load_composed_config(config_path)
    runtime_cfg = config.get("runtime", {})
    signal_files = runtime_cfg.get("signal_files", {}) if isinstance(runtime_cfg, dict) else {}
    resolved = dict(signal_files) if isinstance(signal_files, dict) else {}
    logger.info(f"Loaded signal files from config: {resolved}")
    return resolved


def extract_r1_runtime_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve NEDF3 hardware/control metadata from the active device config."""
    metadata_cfg = config.get("nedf3_metadata", {})
    if not isinstance(metadata_cfg, dict):
        raise ValueError("nedf3_metadata must be a mapping")

    devices_section = config.get("devices", [])
    if isinstance(devices_section, dict):
        devices_section = devices_section.get("devices", [])
    if not isinstance(devices_section, list):
        raise ValueError("devices config must resolve to a list")

    rizon_devices = [
        device
        for device in devices_section
        if isinstance(device, dict)
        and device.get("class") in {"RizonRobot", "RizonRobotLKForceGated"}
    ]
    if len(rizon_devices) != 1:
        raise ValueError(
            "Expected exactly one Rizon robot device for R1 metadata, "
            f"found {len(rizon_devices)}"
        )
    robot_cfg = rizon_devices[0]

    control_cfg = robot_cfg.get("control")
    if not isinstance(control_cfg, dict):
        raise ValueError("Rizon control must be a mapping")
    control_type = control_cfg.get("type")
    if control_type == "joints":
        control_mode = "NRT_JOINT_POSITION"
    elif control_type in {"eef_abs", "eef_rel"}:
        control_mode = "NRT_CARTESIAN_MOTION_FORCE"
    else:
        raise ValueError(f"Unsupported Rizon control.type for metadata: {control_type!r}")

    impedance = control_cfg.get("cartesian_impedance")
    flexiv_impedance = None
    if impedance is not None:
        if not isinstance(impedance, dict):
            raise ValueError("Rizon control.cartesian_impedance must be a mapping")
        translation = impedance.get("translation_n_m")
        rotation = impedance.get("rotation_nm_rad")
        if not isinstance(translation, (list, tuple)) or len(translation) != 3:
            raise ValueError(
                "Rizon control.cartesian_impedance.translation_n_m must contain 3 values"
            )
        if not isinstance(rotation, (list, tuple)) or len(rotation) != 3:
            raise ValueError(
                "Rizon control.cartesian_impedance.rotation_nm_rad must contain 3 values"
            )
        flexiv_impedance = {
            "stiffness": [float(value) for value in (*translation, *rotation)]
        }

    hardware_string = metadata_cfg.get("hardware_string")
    wrench_calibration = metadata_cfg.get("eef_wrench_calibration")
    return {
        "hardware_string": str(hardware_string) if hardware_string not in (None, "") else None,
        "flexiv_control_mode": control_mode,
        "flexiv_impedance": flexiv_impedance,
        "eef_wrench_calibration": wrench_calibration,
    }


# Canonical lab_pro NEDF3 metadata (18 topics); see lab_pro_canonical_metadata_example.json.
NEDF3_RECORD_TOPIC_LIST: Tuple[str, ...] = (
    "nmx/hal/camera/camera_0/rgbd",
    "nmx/hal/umi/right/pose",
    "nmx/hal/umi/right/hand_state",
    "nmx/hal/umi/right/raw_hand_state",
    "nmx/hal/umi/right/slam",
    "nmx/hal/umi/right/extra",
    "nmx/hal/camera/camera_1/rgbd",
    "nmx/hal/umi/left/pose",
    "nmx/hal/umi/left/hand_state",
    "nmx/hal/umi/left/raw_hand_state",
    "nmx/hal/umi/left/slam",
    "nmx/hal/umi/left/extra",
    "nmx/hal/camera/camera_2/rgbd",
    "nmx/hal/umi/ego/pose",
    "nmx/hal/umi/ego/hand_state",
    "nmx/hal/umi/ego/raw_hand_state",
    "nmx/hal/umi/ego/slam",
    "nmx/hal/umi/ego/extra",
)

NEDF3_EXTRA_KEYS: Tuple[str, ...] = (
    "nmx/hal/umi/right/raw_hand_state",
    "nmx/hal/umi/right/slam",
    "nmx/hal/umi/right/extra",
    "nmx/hal/umi/left/raw_hand_state",
    "nmx/hal/umi/left/slam",
    "nmx/hal/umi/left/extra",
    "nmx/hal/umi/ego/raw_hand_state",
    "nmx/hal/umi/ego/slam",
    "nmx/hal/umi/ego/extra",
)

# ViveTracker row index → UMI side name (matches device row layout: left, right, ego).
_VIVE_ROW_TO_UMI_SIDE: Tuple[str, str, str] = ("left", "right", "ego")

HAL_CAMERA_TOPICS: Tuple[str, str, str] = (
    "nmx/hal/camera/camera_0/rgbd",
    "nmx/hal/camera/camera_1/rgbd",
    "nmx/hal/camera/camera_2/rgbd",
)

CAMERA_DEVICE_TYPES: frozenset = frozenset(
    {"OpenCVCameraDevice", "HikCameraDevice", "WebCameraDevice", "WebCameraOpenPi"}
)

# Device types present in the summary SHM that must NOT be recorded into NEDF3.
# ZedStereoDevice is the operator's FPV stream (needed for teleop, but not part
# of the dataset); Quest3VuerTeleoperator publishes the raw EEF command (the
# recorded robot STATE comes from RizonRobot/YamHeadRobot instead). Skipping
# them avoids huge no-op frame copies (ZED ~5 MB/frame) and per-frame warnings.
IGNORED_RECORD_DEVICE_TYPES: frozenset = frozenset(
    {"ZedStereoDevice", "Quest3VuerTeleoperator"}
)

# Metadata ``camera_info`` keys are ``camera_0``..``camera_2``; values are fixed role names
# for device suffix N in ``WebCameraDevice_N`` / ``OpenCVCameraDevice_N`` / ``HikCameraDevice_N``.
NEDF3_CAMERA_ROLE_LABELS: Tuple[str, str, str] = (
    "right_wrist",
    "left_wrist",
    "head",
)

# Stage 1 ``extra.processed`` null placeholders (canonical lab_pro / nmx_processor).
_PROCESSED_EXTRA_FIELDS: Tuple[str, ...] = (
    "aligned_timestamp_ns",
    "alignment_mode",
    "downsample_kept",
    "aligned_pose",
    "aligned_hand_state",
    "aligned_raw_hand_state",
    "aligned_slam",
    "aligned_pose_tcp",
    "aligned_corrected_roi_marker_angle",
    "aligned_corrected_gripper_width",
    "aligned_arkit_diagnostics",
)

def _umi_hal_topic(side: str, kind: str) -> str:
    return f"nmx/hal/umi/{side}/{kind}"


def _pose7_to_umi_pose(row: np.ndarray) -> Any:
    """Build ``UmiPose`` from Vive ``[x,y,z,qx,qy,qz,qw]`` (nmx_msg Pose3 + UmiPose.pose)."""
    from nmx_msg.Quaternion_pb2 import Quaternion
    from nmx_msg.Umi_pb2 import UmiPose
    from nmx_msg.Vec_pb2 import Vec3

    x, y, z, qx, qy, qz, qw = np.asarray(row, dtype=np.float64).reshape(7)
    umi_pose = UmiPose()
    umi_pose.pose.pos.CopyFrom(
        Vec3(x=float(x), y=float(y), z=float(z))
    )
    umi_pose.pose.quat.CopyFrom(
        Quaternion(w=float(qw), x=float(qx), y=float(qy), z=float(qz))
    )
    return umi_pose


def _empty_umi_state() -> Any:
    """Empty ``UmiState`` (no joints) for unavailable hand/raw streams."""
    from nmx_msg.Umi_pb2 import UmiState

    return UmiState()


def _umi_state_from_joints(
    joint_values: Any,
    gripper_width: Optional[float] = None,
    joint_prefix: str = "joint",
) -> Any:
    """Build ``UmiState`` from arm joint angles (+ optional gripper width).

    ``UmiState`` only has a repeated ``joints`` (``JointState{name,pos,...}``)
    field, so robot joint angles and the gripper opening are all carried as
    named ``JointState`` entries (``joint0..jointN`` then ``gripper``).
    """
    from nmx_msg.Umi_pb2 import UmiState

    # NOTE: repeated message fields are populated via ``.joints.add()``; the
    # nested JointState class does not need to be imported explicitly.
    state = UmiState()
    for i, pos in enumerate(np.asarray(joint_values, dtype=np.float64).reshape(-1)):
        js = state.joints.add()
        js.name = f"{joint_prefix}{i}"
        js.pos = float(pos)
    if gripper_width is not None:
        jg = state.joints.add()
        jg.name = "gripper"
        jg.pos = float(gripper_width)
    return state


def _flexiv_tcp_to_pose7(tcp: Any) -> np.ndarray:
    """Flexiv ``tcp_pose`` [x,y,z,qw,qx,qy,qz] -> NEDF pose7 [x,y,z,qx,qy,qz,qw]."""
    t = np.asarray(tcp, dtype=np.float64).reshape(-1)
    x, y, z, qw, qx, qy, qz = t[:7]
    return np.array([x, y, z, qx, qy, qz, qw], dtype=np.float64)


def _empty_slam_frame_exceptions() -> Any:
    """Empty SLAM payload: ``UmiSlamFrameExceptions`` with ``exceptions`` length 0.

    UMI / lab_pro schema only -- the R1 saver does NOT record any ``slam`` topic
    (see ``R1_RECORD_TOPIC_LIST``), so this is never called on the R1 path. The
    previous vendored fallback (``_vendor/UmiSlam_pb2.py``) was removed; if the
    UMI path is ever used on a build whose ``nmx_msg`` omits ``UmiSlam_pb2``,
    this raises a clear ``ModuleNotFoundError``.
    """
    from nmx_msg.UmiSlam_pb2 import UmiSlamFrameExceptions

    return UmiSlamFrameExceptions()


def _empty_processed_extra_payload() -> Dict[str, Any]:
    return {key: None for key in _PROCESSED_EXTRA_FIELDS}


def _empty_extra_metadata_message() -> Any:
    """``extra`` HAL topic payload with Stage 1 ``processed`` null placeholders."""
    from nmx_msg.Metadata_pb2 import Metadata

    meta_msg = Metadata()
    meta_msg.version = "nedf3"
    payload = {"processed": _empty_processed_extra_payload()}
    meta_msg.metadata = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return meta_msg


def _extra_metadata_message(payload: Dict[str, Any]) -> Any:
    """Build an ``extra`` topic message containing recorder-owned raw data."""
    from nmx_msg.Metadata_pb2 import Metadata

    meta_msg = Metadata()
    meta_msg.version = "nedf3"
    meta_msg.metadata = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return meta_msg


def _build_encoded_image_message(
    data: bytes, rows: int, cols: int, channels: int, encoded_format: str
) -> Image:
    image = Image()
    image.data = data
    image.type = ImageDataType.ENCODED
    image.encoded_format = encoded_format
    image.rows = int(rows)
    image.cols = int(cols)
    image.channels = int(channels)
    return image


def _empty_depth_image_message() -> Image:
    """Canonical empty/invalid depth: zero dims, empty encoded_format, no payload."""
    return _build_encoded_image_message(b"", 0, 0, 0, "")


def _build_rgbd_from_jpeg(
    jpg_bytes: bytes, rows: int, cols: int, channels: int
) -> RGBD:
    """Wrap JPEG RGB bytes in ``RGBD`` with empty depth (no depth sensor on collect path)."""
    rgbd = RGBD()
    rgbd.rgb.CopyFrom(
        _build_encoded_image_message(jpg_bytes, rows, cols, channels, "jpeg")
    )
    rgbd.depth.CopyFrom(_empty_depth_image_message())
    if hasattr(rgbd, "depth_scale"):
        rgbd.depth_scale = 1.0
    return rgbd


# Wrist (left/right) stored-image geometry: center-crop the native frame to the
# pretraining aspect ratio (h x w = 360 x 270, i.e. 3:4), then resize up so the
# pixel grid matches the head camera (h x w = 640 x 480). Aspect ratio is preserved
# (270/360 == 480/640), so the resize is distortion-free.
NEDF3_WRIST_CROP_H: int = 360             # center-crop height (pixels)
NEDF3_WRIST_CROP_W: int = 270             # center-crop width (pixels)
NEDF3_WRIST_OUT_H: int = 640              # final stored height (pixels)
NEDF3_WRIST_OUT_W: int = 480              # final stored width (pixels)


def _center_crop_and_resize_image(
    image: np.ndarray,
    crop_h: int = NEDF3_WRIST_CROP_H,
    crop_w: int = NEDF3_WRIST_CROP_W,
    out_h: int = NEDF3_WRIST_OUT_H,
    out_w: int = NEDF3_WRIST_OUT_W,
) -> np.ndarray:
    """Center-crop ``image`` to ``crop_h`` x ``crop_w`` then resize to ``out_h`` x ``out_w``.

    Used for wrist cameras: a 360x640 (h x w) frame is center-cropped to 360x270
    (keeping full height, trimming width to the 3:4 pretraining aspect) and resized
    up to 640x480. Robust to inputs smaller than the crop (crops what is available).
    """
    if image is None or image.ndim < 2:
        return image

    h, w = image.shape[:2]
    y0 = max(0, (h - int(crop_h)) // 2)
    x0 = max(0, (w - int(crop_w)) // 2)
    cropped = image[y0:y0 + int(crop_h), x0:x0 + int(crop_w)]
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        return image

    return cv2.resize(cropped, (int(out_w), int(out_h)), interpolation=cv2.INTER_LINEAR)


def _camera_role_from_id(camera_id: Optional[str]) -> str:
    """Map a canonical ``camera_N`` id to its role label (e.g. ``head``/``left_wrist``)."""
    try:
        idx = int(str(camera_id).rsplit("_", 1)[-1])
        if 0 <= idx < len(NEDF3_CAMERA_ROLE_LABELS):
            return NEDF3_CAMERA_ROLE_LABELS[idx]
    except (ValueError, AttributeError, TypeError):
        pass
    return ""


def _process_camera_image(image: np.ndarray, camera_id: Optional[str]) -> np.ndarray:
    """Apply the per-role stored-image transform.

    - head: kept as-is (native 480x640, i.e. 3:4 height:width); no crop, pad or resize.
    - left/right wrist: center-crop to the pretraining aspect and resize up
      (360x640 -> crop 360x270 -> resize 640x480).
    - unknown role: returned unchanged.
    """
    role = _camera_role_from_id(camera_id)
    if "wrist" in role:
        return _center_crop_and_resize_image(image)
    return image


try:
    # Preferred when launched from the repo root (this file now lives there).
    from consumer.data_saver import DataSaverConsumer as SyncDataSaverConsumer
    from consumer.data_saver import wait_enter_blocking, enter_pressed_nonblocking
except ImportError:
    try:
        from .data_saver import DataSaverConsumer as SyncDataSaverConsumer
        from .data_saver import wait_enter_blocking, enter_pressed_nonblocking
    except ImportError:
        from data_saver import DataSaverConsumer as SyncDataSaverConsumer
        from data_saver import wait_enter_blocking, enter_pressed_nonblocking


class DataSaverConsumer(AsyncWriteLifecycleMixin, SyncDataSaverConsumer):
    """
    Data saver consumer for storing device manager data to NEDF3 (UMI) format.

    This consumer saves data using NEDF SDK:
    - Camera devices: canonical ids ``camera_0``..``camera_2``; JPEG in HAL ``RGBD`` messages (no intrinsic/extrinsic MCAP topics)
    - Vive: rows of 7D → ``nmx/hal/umi/{left,right,ego}/pose``; per-frame empty ``hand_state`` /
      ``raw_hand_state`` / ``slam`` / ``extra`` when no hardware source (no fake zeros)
    - Metadata: canonical 18-topic ``record_topic_list`` and ``extra_keys`` (lab_pro layout)

    Features:
    - Async writing for high-frequency data collection
    - Support for image resizing
    - NEDF3 metadata generation (no I2RT sidecar commands)
    """

    def __init__(self, summary_shm_name: str = "device_summary_data",
                 output_dir: str = "saved_data", save_interval: float = 0.04,
                 start_idx: int = 0, image_size: Optional[Tuple[int, int]] = None,
                 async_write: bool = True, queue_size: int = 5000,
                 writer_shutdown_timeout_s: float = 30.0,
                 robot_type: str = "UMI", task_name: str = "data_collection",
                 task_description: str = "",
                 device_identifier: str = "",
                 user_id: str = "",
                 user_name: str = "",
                 end_effector_vendor_model: str = "",
                 umi_to_tcp_transform: Optional[List[float]] = None,
                 device_type: str = "teleoperation_r1",
                 hardware_string: Optional[str] = None,
                 flexiv_control_mode: Optional[str] = None,
                 flexiv_impedance: Optional[Dict[str, Any]] = None,
                 eef_wrench_calibration: Optional[Any] = None,
                 enable_anomaly_detection: bool = True,
                 anomaly_config_path: Optional[str] = None,
                 enable_button_control: bool = True,
                 signal_files: Optional[Dict[str, str]] = None) -> None:
        """
        Initialize the data saver consumer.

        Args:
            summary_shm_name: Name of the summary shared memory
            output_dir: Directory to save data files
            save_interval: Interval between saves in seconds
            start_idx: Starting episode index
            image_size: Target image size (width, height) for resizing
            async_write: Whether to use async writing thread (default: True)
            queue_size: Size of async write queue (default: 5000)
            writer_shutdown_timeout_s: Maximum time to drain the async queue on shutdown
            robot_type: Value for NEDF ``robot_vendor_model`` (e.g. UMI or Vendor_Model)
            task_name: Task name for NEDF metadata
            end_effector_vendor_model: NEDF3 ``end_effector_vendor_model`` (VendorName_ModelName per spec)
            umi_to_tcp_transform: NEDF3 ``umi_to_tcp_transform`` (row-major 4x4 floats per arm).
                If omitted (``None``), stored as ``[]`` in metadata.
            enable_anomaly_detection: Whether to enable episode anomaly detection
            anomaly_config_path: Path to anomaly detection config file
        """
        super().__init__(summary_shm_name, output_dir, save_interval, start_idx, image_size)

        # Async writing infrastructure
        self.async_write = async_write
        self._initialize_async_write_lifecycle(
            queue_size,
            writer_shutdown_timeout_s,
        )
        self._episode_write_failed = False
        self._episode_write_failure_reason: Optional[str] = None

        # NEDF3 writer management
        self.nedf_writer = None
        self.nedf_writer_context = None  # Store the context manager
        self.topics_config: Optional[NEDFTopicsConfig] = None
        self.current_episode_dir = None
        self.episode_active = False
        self.device_data_counts: Dict[str, int] = {}

        # NEDF metadata configuration
        self.robot_type = robot_type
        self.task_name = task_name
        # Free-form task description written into NEDFMetadata.task_description.
        self.task_description = task_description or ""
        # metadata.json extra fields (GUI-fillable). ``device`` is a list of
        # {"identifier": ...} per the R1 NEDF3 schema; user_id/user_name tag the
        # operator who recorded the episode.
        self.user_id = user_id or ""
        self.user_name = user_name or ""
        self.device_identifier = device_identifier or ""
        self.end_effector_vendor_model = end_effector_vendor_model
        # Top-level metadata.json "device_type" (injected post-write since
        # NEDFMetadata ignores extra fields).
        self.device_type = device_type
        self.hardware_string = hardware_string
        self.flexiv_control_mode = flexiv_control_mode
        self.flexiv_impedance = dict(flexiv_impedance) if flexiv_impedance is not None else None
        self.eef_wrench_calibration = eef_wrench_calibration
        self.umi_to_tcp_transform = (
            list(umi_to_tcp_transform) if umi_to_tcp_transform is not None else []
        )

        # Camera ID mapping (device_key -> camera_id)
        self.camera_id_mapping: Dict[str, str] = {}
        # Camera topic mapping (camera_id -> image topic)
        self.camera_image_topic_mapping: Dict[str, str] = {}

        # Episode anomaly detection
        self.enable_anomaly_detection = enable_anomaly_detection
        self.anomaly_config_path = anomaly_config_path
        self.anomaly_detector = None
        self.anomaly_stats = {
            'total_episodes_checked': 0,
            'total_anomalies': 0,
            'episodes_with_errors': 0,
            'last_check_time': 0
        }
        # Episode signal files (optional; I2RT robot command patterns are not used)
        if signal_files is None:
            signal_files = {}

        # Button control for episode management
        self.enable_button_control = enable_button_control
        episode_signal_pattern = signal_files.get("episode_signal_pattern")
        if self.enable_button_control:
            if not episode_signal_pattern:
                raise ValueError(
                    "signal_files.episode_signal_pattern is required when enable_button_control is True "
                    "(pass --config with runtime.signal_files, or provide signal_files in code)."
                )
            self.episode_signal_pattern = episode_signal_pattern
        else:
            # Placeholder paths when using Enter-based control only
            self.episode_signal_pattern = episode_signal_pattern or "/tmp/nedf3_episode_unused_{signal}.signal"

        self._episode_start_signal_path = self.episode_signal_pattern.format(signal="start")
        self._episode_end_signal_path = self.episode_signal_pattern.format(signal="end")
        self.episode_running = False
        self._discard_keyboard_pressed = False

        # Keyboard listener for manual episode discard (pressing 'd')
        self.keyboard_listener = None

        # GUI control mode: when enabled, episode start/stop/discard is driven by
        # threading events set from the Tk GUI instead of terminal Enter or the
        # leader-robot button signals. See start_saving() and the gui_* methods.
        self.gui_control = False
        self._gui_start_event = threading.Event()
        self._gui_stop_event = threading.Event()
        self._gui_discard_event = threading.Event()
        # Path of the most recently finalized episode (for GUI replay/delete).
        self.last_finalized_episode_dir: Optional[str] = None

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Load anomaly detection configuration
        if self.enable_anomaly_detection:
            self._load_anomaly_config()

    def _load_anomaly_config(self) -> None:
        """Load anomaly detection configuration from file."""
        try:
            if self.anomaly_config_path:
                config_path = Path(self.anomaly_config_path).expanduser()
                if not config_path.exists():
                    raise FileNotFoundError(
                        f"Anomaly detection config does not exist: {config_path}"
                    )
            else:
                config_path = PROJECT_ROOT / "configs" / "anomaly_config" / "default.yaml"

                if not config_path.exists():
                    logger.error("No anomaly detection config file found, disabling anomaly detection")
                    self.enable_anomaly_detection = False
                    return

            with config_path.open("r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
                self.anomaly_config = full_config.get('anomaly_detection', {})

            if not self.anomaly_config:
                logger.warning("No 'anomaly_detection' section found in config file, disabling anomaly detection")
                self.enable_anomaly_detection = False
                return

            # Initialize anomaly detector
            self.anomaly_detector = EpisodeAnomalyDetector(self.anomaly_config)
            logger.info(f"Loaded anomaly detection config from: {config_path}")

        except Exception as e:
            logger.error(f"Error loading anomaly detection config: {e}")
            logger.warning("Disabling anomaly detection due to config error")
            self.enable_anomaly_detection = False

    def _detect_episode_anomalies(self, episode_dir: str) -> None:
        """
        Detect anomalies in the saved episode.

        Args:
            episode_dir: Path to the episode directory
        """
        if not self.enable_anomaly_detection or not self.anomaly_detector:
            return

        try:
            episode_path = Path(episode_dir)

            feat_mapping = self._episode_anomaly_feature_mapping()

            # Run anomaly detection
            anomalies, has_errors = self.anomaly_detector.detect_episode_anomalies(episode_path, feat_mapping)

            # Update statistics
            self.anomaly_stats['total_episodes_checked'] += 1
            self.anomaly_stats['total_anomalies'] += len(anomalies)
            self.anomaly_stats['last_check_time'] = time.time()

            if has_errors:
                self.anomaly_stats['episodes_with_errors'] += 1

            # Log anomalies
            if anomalies:
                logger.warning(f"🔍 Episode {episode_path.name}: Found {len(anomalies)} anomalies")

                errors = [a for a in anomalies if a['severity'] == 'error']
                warnings = [a for a in anomalies if a['severity'] == 'warning']

                if errors:
                    logger.error(f"  ❌ {len(errors)} errors detected:")
                    for error in errors:
                        logger.error(f"    - {error['message']}")

                if warnings:
                    logger.warning(f"  ⚠️  {len(warnings)} warnings detected:")
                    for warning in warnings:
                        logger.warning(f"    - {warning['message']}")
            else:
                logger.info(f"✅ Episode {episode_path.name}: No anomalies detected")

        except Exception as e:
            logger.error(f"Error detecting anomalies in episode {episode_dir}: {e}")
            import traceback
            traceback.print_exc()

    def _episode_anomaly_feature_mapping(self) -> Dict[str, str]:
        """Build detector inputs from the recorder's canonical device mapping."""
        feat_mapping: Dict[str, str] = {}
        for device_key, device_info in self.devices.items():
            device_type = device_info["type"]
            if device_type in CAMERA_DEVICE_TYPES:
                camera_id = self.camera_id_mapping.get(device_key)
                cam_idx = (
                    self._camera_index_from_device_key(camera_id)
                    if camera_id
                    else None
                )
                if cam_idx == 0:
                    feat_mapping["img_right"] = device_key
                elif cam_idx == 1:
                    feat_mapping["img_left"] = device_key
                elif cam_idx == 2:
                    feat_mapping["img_ego"] = device_key
            elif device_type == "ViveTrackerDevice":
                feat_mapping["pose_unified"] = device_key
        return feat_mapping

    def _setup_keyboard_listener(self) -> None:
        """Setup keyboard listener for manual episode discard using 'd' key."""
        if pynput_keyboard is None:
            logger.warning(
                "Keyboard listener is unavailable; use the control UI to discard episodes."
            )
            self.keyboard_listener = None
            return
        try:

            def on_press(key: Any) -> None:
                try:
                    if hasattr(key, 'char') and key.char == 'd' and self.episode_running:
                        self._discard_keyboard_pressed = True
                        logger.info("'D' pressed - discarding episode and resetting to home")
                except AttributeError:
                    pass

            self.keyboard_listener = pynput_keyboard.Listener(on_press=on_press)
            self.keyboard_listener.start()
            logger.info("Keyboard listener started. Press 'D' to discard current episode and reset to home.")
        except Exception as e:
            logger.error(f"Failed to setup keyboard listener: {e}")
            self.keyboard_listener = None

    def _stop_keyboard_listener(self) -> None:
        """Stop the keyboard listener."""
        if self.keyboard_listener is not None:
            try:
                self.keyboard_listener.stop()
                self.keyboard_listener = None
                logger.info("Keyboard listener stopped")
            except Exception as e:
                logger.warning(f"Failed to stop keyboard listener: {e}")



    # ------------------------------------------------------------------ #
    # GUI control hooks (used when gui_control=True)
    # ------------------------------------------------------------------ #
    def set_metadata_fields(self, task_name: Optional[str] = None,
                            task_description: Optional[str] = None,
                            device_identifier: Optional[str] = None,
                            user_id: Optional[str] = None,
                            user_name: Optional[str] = None) -> None:
        """Update GUI-fillable metadata fields (applied to the next episode)."""
        if task_name is not None:
            self.task_name = task_name
        if task_description is not None:
            self.task_description = task_description
        if device_identifier is not None:
            self.device_identifier = device_identifier
        if user_id is not None:
            self.user_id = user_id
        if user_name is not None:
            self.user_name = user_name

    def gui_start_episode(self) -> None:
        """Signal the saving loop to begin a new episode (GUI mode)."""
        self._gui_stop_event.clear()
        self._gui_discard_event.clear()
        self._gui_start_event.set()

    def gui_stop_episode(self) -> None:
        """Signal the saving loop to finalize the current episode (GUI mode)."""
        self._gui_stop_event.set()

    def gui_discard_episode(self) -> None:
        """Signal the saving loop to discard the current episode (GUI mode)."""
        self._gui_discard_event.set()
        self._gui_stop_event.set()

    def request_shutdown(self) -> None:
        """Ask the saving loop to exit cleanly (GUI window closed)."""
        self.running = False
        # Unblock any wait on the start event so the outer loop can exit.
        self._gui_start_event.set()
        self._gui_stop_event.set()

    def _gui_wait_for_start(self) -> bool:
        """Block until the GUI requests an episode start. Returns False on shutdown."""
        self._gui_start_event.clear()
        while self.running:
            if self._gui_start_event.wait(timeout=0.1):
                self._gui_start_event.clear()
                return self.running
        return False

    def _wait_for_episode_start_signal(self) -> None:
        """Wait for episode start signal file to be created by leader robot."""
        while self.running and not os.path.exists(self._episode_start_signal_path):
            time.sleep(0.01)
        # Clear the signal file after detection
        self._clear_episode_signal("start")

    def _check_episode_end_signal(self) -> bool:
        """Check if episode end signal file exists."""
        if os.path.exists(self._episode_end_signal_path):
            self._clear_episode_signal("end")
            return True
        return False

    def _clear_episode_signal(self, signal_type: str) -> None:
        """Clear the episode signal file after processing."""
        signal_path = self.episode_signal_pattern.format(signal=signal_type)
        try:
            if os.path.exists(signal_path):
                os.remove(signal_path)
                logger.debug(f"Cleared episode {signal_type} signal")
        except Exception as e:
            logger.warning(f"Failed to clear episode {signal_type} signal: {e}")

    def open_episode_streams(self, episode_dir: str) -> None:
        """
        Open NEDF3 writer for the current episode.

        Args:
            episode_dir: Directory path for this episode
        """
        try:
            os.makedirs(episode_dir, exist_ok=True)
            self.current_episode_dir = episode_dir
            self._reset_episode_write_state()
            self.topics_config = NEDFTopicsConfig()

            # Build camera_info: keys camera_0..camera_2, values role labels; map device_key -> canonical id
            camera_entries = self._camera_entries()
            camera_info: Dict[str, str] = {}
            for cam_idx, device_key, device_info in camera_entries:
                canonical_id = f"camera_{cam_idx}"
                if cam_idx < len(NEDF3_CAMERA_ROLE_LABELS):
                    role_label = NEDF3_CAMERA_ROLE_LABELS[cam_idx]
                else:
                    role_label = str(device_info.get("name", canonical_id))
                camera_info[canonical_id] = role_label
                self.camera_id_mapping[device_key] = canonical_id
                self.camera_image_topic_mapping[canonical_id] = (
                    self._resolve_camera_image_topic(canonical_id)
                )
                logger.debug(
                    f"Registered camera: {device_key} -> {canonical_id} ({role_label}), "
                    f"image_topic={self.camera_image_topic_mapping[canonical_id]}"
                )

            main_camera = (
                "camera_0"
                if "camera_0" in camera_info
                else (next(iter(camera_info)) if camera_info else "CAMID_1")
            )
            n_cam = len(camera_info)
            arm_layout = "dual_arm" if n_cam >= 2 else "single_arm"

            record_topic_list = list(NEDF3_RECORD_TOPIC_LIST)

            metadata = NEDFMetadata(
                hardware_config_name="",
                task_name=self.task_name,
                task_description=self.task_description,
                robot_vendor_model=self.robot_type,
                data_format="NEDF3",
                mcap_file_name="data.mcap",
                mcap_file_type="CHUNK_FILE",
                collection_mode="teleoperation",
                end_effector="gripper",
                end_effector_vendor_model=self.end_effector_vendor_model,
                robot_type=arm_layout,
                arm_base_to_world_transform=[],
                umi_to_tcp_transform=self.umi_to_tcp_transform,
                camera_info=camera_info,
                main_camera=main_camera,
                extra_keys=list(NEDF3_EXTRA_KEYS),
                extra={},
                record_topic_list=record_topic_list,
            )

            metadata_path = os.path.join(episode_dir, "metadata.json")
            writer_config = NEDFWriterConfig(
                metadata_file_path=Path(metadata_path),
                metadata=metadata,
                topics_config=self.topics_config,
                max_file_size_mb=100,
                mcap_chunk_size_kb=256,
            )

            self.nedf_writer_context = NEDFFactory.get_writer(NEDFType.NEDF3, writer_config)
            self.nedf_writer = self.nedf_writer_context.__enter__()

            for device_key in self.devices:
                self.device_data_counts[device_key] = 0

            self.episode_active = True
            logger.info(f"Opened NEDF3 writer in {episode_dir}")
            logger.info(f"Cameras: {list(camera_info.keys())}")
            logger.info(f"record_topic_list ({len(record_topic_list)}): {record_topic_list}")

        except Exception as e:
            logger.error(f"Error opening NEDF3 writer: {e}")
            import traceback
            traceback.print_exc()
            raise

    def close_episode_streams(self) -> None:
        """Close NEDF3 writer."""
        try:
            if self.nedf_writer_context is not None:
                self.nedf_writer_context.__exit__(None, None, None)
                self.nedf_writer = None
                self.nedf_writer_context = None
                self.topics_config = None

                for device_key, count in self.device_data_counts.items():
                    logger.debug(f"Wrote {count} frames for {device_key}")

                logger.info("Closed NEDF3 writer")

            self.device_data_counts.clear()
            self.camera_id_mapping.clear()
            self.camera_image_topic_mapping.clear()
            self.episode_active = False

        except Exception as e:
            logger.error(f"Error closing NEDF3 writer: {e}")
            import traceback
            traceback.print_exc()

    def _write_canonical_umi_companion_topics(
        self, side: str, timestamp_ns: int
    ) -> None:
        """Write empty/null companion UMI topics for one side (no fake numeric placeholders)."""
        writer = self.nedf_writer
        if writer is None:
            return
        writer.write_message(
            _umi_hal_topic(side, "hand_state"),
            _empty_umi_state(),
            timestamp_ns,
            timestamp_ns,
        )
        writer.write_message(
            _umi_hal_topic(side, "raw_hand_state"),
            _empty_umi_state(),
            timestamp_ns,
            timestamp_ns,
        )
        writer.write_message(
            _umi_hal_topic(side, "slam"),
            _empty_slam_frame_exceptions(),
            timestamp_ns,
            timestamp_ns,
        )
        writer.write_message(
            _umi_hal_topic(side, "extra"),
            _empty_extra_metadata_message(),
            timestamp_ns,
            timestamp_ns,
        )

    def _write_canonical_umi_pose_bundle(
        self, side: str, pose_row: np.ndarray, timestamp_ns: int
    ) -> None:
        """Write real ``pose`` plus empty companion topics for this timestamp."""
        writer = self.nedf_writer
        if writer is None:
            return
        writer.write_message(
            _umi_hal_topic(side, "pose"),
            _pose7_to_umi_pose(pose_row),
            timestamp_ns,
            timestamp_ns,
        )
        self._write_canonical_umi_companion_topics(side, timestamp_ns)

    def _write_umi_robot_bundle(
        self,
        side: str,
        pose7: np.ndarray,
        gripper_width_m: Optional[float],
        gripper_radian: Optional[float],
        timestamp_ns: int,
    ) -> None:
        """Write one UMI side from robot data: real ``pose`` + gripper-only states.

        ``pose``           <- EEF Cartesian pose (7D [x,y,z,qx,qy,qz,qw]);
        ``hand_state``     <- gripper physical width (m) as a single ``gripper``
                              ``JointState``, or EMPTY when the side has no gripper
                              (e.g. the YAM ego head arm);
        ``raw_hand_state`` <- gripper motor angle (radian) as a single ``gripper``
                              ``JointState``, or EMPTY when no gripper;
        ``slam`` / ``extra`` <- canonical empty placeholders.

        Arms are recorded as Cartesian pose only (no arm joint angles).
        """
        writer = self.nedf_writer
        if writer is None:
            return
        writer.write_message(
            _umi_hal_topic(side, "pose"),
            _pose7_to_umi_pose(np.asarray(pose7, dtype=np.float64).reshape(-1)),
            timestamp_ns,
            timestamp_ns,
        )
        hand_state_msg = (
            _umi_state_from_joints(np.empty(0, dtype=np.float64), gripper_width=gripper_width_m)
            if gripper_width_m is not None
            else _empty_umi_state()
        )
        raw_hand_state_msg = (
            _umi_state_from_joints(np.empty(0, dtype=np.float64), gripper_width=gripper_radian)
            if gripper_radian is not None
            else _empty_umi_state()
        )
        writer.write_message(
            _umi_hal_topic(side, "hand_state"),
            hand_state_msg,
            timestamp_ns,
            timestamp_ns,
        )
        writer.write_message(
            _umi_hal_topic(side, "raw_hand_state"),
            raw_hand_state_msg,
            timestamp_ns,
            timestamp_ns,
        )
        writer.write_message(
            _umi_hal_topic(side, "slam"),
            _empty_slam_frame_exceptions(),
            timestamp_ns,
            timestamp_ns,
        )
        writer.write_message(
            _umi_hal_topic(side, "extra"),
            _empty_extra_metadata_message(),
            timestamp_ns,
            timestamp_ns,
        )

    def _async_write_worker(self) -> None:
        """
        Background worker thread that continuously consumes data from write queue.

        This thread runs independently, writing data to NEDF3 as it becomes available.
        """
        logger.info("Async write worker thread started")

        while True:
            write_task = None
            try:
                write_task = self._get_next_write_task()
                if write_task is None:
                    break

                # Extract task data
                device_key = write_task['device_key']
                device_type = write_task['device_type']
                timestamp_ns = write_task['timestamp_ns']
                data_array = write_task['data_array']

                # Check if episode is active and writer is open
                if not self.episode_active or self.nedf_writer is None or self.topics_config is None:
                    logger.warning(f"Episode not active or writer not open for {device_key}, dropping data")
                    self.write_queue.task_done()
                    continue

                # Process based on device type
                if device_type in CAMERA_DEVICE_TYPES:
                    # Handle camera data
                    camera_id = self.camera_id_mapping.get(device_key)
                    if camera_id is None:
                        logger.warning(f"Camera ID not found for {device_key}")
                        self.write_queue.task_done()
                        continue

                    # Resize image if needed (head camera is kept fully untouched)
                    if self.image_size is not None and _camera_role_from_id(camera_id) != "head":
                        target_width, target_height = self.image_size
                        original_shape = data_array.shape
                        data_array = cv2.resize(data_array, (target_width, target_height),
                                                interpolation=cv2.INTER_LINEAR)
                        logger.debug(f"Resized image from {original_shape[:2]} to {data_array.shape[:2]}")

                    # Per-role stored geometry: head -> untouched (native frame);
                    # wrist -> aspect crop + resize up to the head pixel grid.
                    data_array = _process_camera_image(data_array, camera_id)

                    # Convert RGB to BGR for OpenCV encoding
                    # Images from devices are stored in RGB format, but cv2.imencode expects BGR
                    if len(data_array.shape) == 3 and data_array.shape[2] == 3:
                        data_array = cv2.cvtColor(data_array, cv2.COLOR_RGB2BGR)

                    # Encode image as JPG
                    _, jpg_bytes = cv2.imencode('.jpg', data_array, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    rows = int(data_array.shape[0])
                    cols = int(data_array.shape[1])
                    channels = int(data_array.shape[2]) if len(data_array.shape) == 3 else 1
                    rgbd_msg = _build_rgbd_from_jpeg(
                        jpg_bytes.tobytes(), rows, cols, channels
                    )

                    rgbd_topic = self.camera_image_topic_mapping.get(camera_id)
                    if not rgbd_topic:
                        rgbd_topic = self._resolve_camera_image_topic(camera_id)
                    self.nedf_writer.write_message(
                        rgbd_topic, rgbd_msg, timestamp_ns, timestamp_ns
                    )

                elif device_type == "ViveTrackerDevice":
                    # (N, 7) rows: [0]→left, [1]→right, [2]→ego; pose + empty companion topics.
                    flat = np.asarray(data_array, dtype=np.float64).reshape(-1)
                    n = flat.size // 7
                    if n <= 0:
                        logger.warning(
                            f"ViveTrackerDevice {device_key}: need at least 7 floats, got {flat.size}"
                        )
                        self.write_queue.task_done()
                        continue
                    if flat.size % 7 != 0:
                        logger.debug(
                            f"ViveTrackerDevice {device_key}: length {flat.size} not a multiple of 7, truncating"
                        )
                    n_write = min(n, len(_VIVE_ROW_TO_UMI_SIDE))
                    mat = flat[: n_write * 7].reshape(n_write, 7)
                    for i in range(n_write):
                        side = _VIVE_ROW_TO_UMI_SIDE[i]
                        self._write_canonical_umi_pose_bundle(
                            side, mat[i], timestamp_ns
                        )

                elif "RizonRobot" in device_type:
                    # Dual Flexiv state. Canonical [44]: per arm q(7) dq(7)
                    # tcp_pose(7) grip_norm(1); left=[0:22], right=[22:44]; tcp_pose
                    # is [x,y,z,qw,qx,qy,qz]. NEDF3 collection adds a 4-wide tail
                    # [48] = [L_grip_width_m, L_grip_rad, R_grip_width_m, R_grip_rad].
                    flat = np.asarray(data_array, dtype=np.float64).reshape(-1)
                    if flat.size < RizonDualArmStateLayout.BASE_SIZE:
                        logger.warning(
                            f"{device_type} {device_key}: expected >=44 values, got {flat.size}"
                        )
                        self.write_queue.task_done()
                        continue
                    if flat.size >= RizonDualArmStateLayout.GRIPPER_STATE_SIZE:
                        l_width = float(flat[RizonDualArmStateLayout.LEFT_GRIPPER_WIDTH_M])
                        l_rad = float(flat[RizonDualArmStateLayout.LEFT_GRIPPER_RADIAN])
                        r_width = float(flat[RizonDualArmStateLayout.RIGHT_GRIPPER_WIDTH_M])
                        r_rad = float(flat[RizonDualArmStateLayout.RIGHT_GRIPPER_RADIAN])
                    else:
                        # Teleop-shaped SHM: only the normalized gripper value is
                        # available (no width/radian calibration). Record it as
                        # hand_state and leave raw_hand_state (radian) empty.
                        logger.debug(
                            f"{device_type} {device_key}: size {flat.size} < 48; "
                            f"using normalized gripper for hand_state, radian empty"
                        )
                        l_width = float(flat[RizonDualArmStateLayout.LEFT_GRIPPER_VALUE])
                        l_rad = None
                        r_width = float(flat[RizonDualArmStateLayout.RIGHT_GRIPPER_VALUE])
                        r_rad = None
                    # LEFT arm -> umi/left, RIGHT arm -> umi/right (matches camera
                    # roles camera_1=left_wrist, camera_0=right_wrist).
                    self._write_umi_robot_bundle(
                        "left",
                        _flexiv_tcp_to_pose7(
                            flat[RizonDualArmStateLayout.LEFT_MEASURED_TCP]
                        ),
                        l_width,
                        l_rad,
                        timestamp_ns,
                    )
                    self._write_umi_robot_bundle(
                        "right",
                        _flexiv_tcp_to_pose7(
                            flat[RizonDualArmStateLayout.RIGHT_MEASURED_TCP]
                        ),
                        r_width,
                        r_rad,
                        timestamp_ns,
                    )

                elif device_type == "YamHeadRobot":
                    # [13] = q(6) + ee_pose(7, [x,y,z,qx,qy,qz,qw]); ego slot.
                    # YAM has NO gripper -> ego/hand_state + raw_hand_state stay
                    # EMPTY; only ego/pose (FK) is recorded.
                    flat = np.asarray(data_array, dtype=np.float64).reshape(-1)
                    if flat.size >= 13:
                        ee_pose7 = flat[6:13]
                    elif flat.size >= 6:
                        # Joints only (teleop-shaped SHM): identity placeholder pose.
                        ee_pose7 = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64)
                    else:
                        logger.warning(
                            f"{device_type} {device_key}: expected >=6 values, got {flat.size}"
                        )
                        self.write_queue.task_done()
                        continue
                    self._write_umi_robot_bundle(
                        "ego", ee_pose7, None, None, timestamp_ns
                    )

                else:
                    logger.warning(f"Unknown device type: {device_type} for {device_key}")
                    # Still mark task as done to avoid queue blocking
                    self.write_queue.task_done()
                    continue

                # Update counter (initialize if not exists)
                if device_key not in self.device_data_counts:
                    self.device_data_counts[device_key] = 0
                self.device_data_counts[device_key] += 1

                # Log progress periodically
                if self.device_data_counts[device_key] % 100 == 0:
                    logger.debug(f"{device_key}: wrote {self.device_data_counts[device_key]} frames")

                self.write_queue.task_done()

            except queue.Empty:
                # No data available, continue waiting
                continue
            except Exception as e:
                if write_task is not None:
                    self.write_queue.task_done()
                self._record_episode_write_failure(f"Async write worker error: {e}")
                logger.error(f"Error in async write worker: {e}")
                import traceback
                traceback.print_exc()

        self.write_thread_running = False
        logger.info("Async write worker thread stopped")

    def _reset_episode_write_state(self) -> None:
        """Clear write-failure state when a new episode starts."""
        self._episode_write_failed = False
        self._episode_write_failure_reason = None

    def _record_episode_write_failure(self, reason: str) -> None:
        """Record a failure that prevents the current episode being finalized."""
        self._episode_write_failed = True
        if self._episode_write_failure_reason is None:
            self._episode_write_failure_reason = reason

    def _mark_episode_incomplete(self, reason: str) -> None:
        """Write an explicit marker while preserving the temporary episode directory."""
        episode_dir = self.current_episode_dir
        if not episode_dir:
            logger.error(f"Incomplete episode without an active directory: {reason}")
            return
        try:
            marker_path = os.path.join(episode_dir, _INCOMPLETE_EPISODE_MARKER)
            marker = {
                "status": "incomplete",
                "reason": reason,
                "marked_at_unix_s": time.time(),
                "unfinished_queue_items": self.write_queue.unfinished_tasks,
            }
            with open(marker_path, "w", encoding="utf-8") as marker_file:
                json.dump(marker, marker_file, ensure_ascii=False, indent=2)
            logger.error(f"Marked episode incomplete: {marker_path}")
        except Exception as exc:
            logger.error(f"Failed to mark episode incomplete: {exc}")

    def _on_async_write_shutdown_failure(self, reason: str) -> None:
        """Preserve R1 episodes that cannot be drained cleanly."""
        self._record_episode_write_failure(reason)
        self._mark_episode_incomplete(reason)

    @staticmethod
    def _camera_index_from_device_key(device_key: str) -> Optional[int]:
        """Parse trailing integer from e.g. ``WebCameraDevice_0`` -> ``0``."""
        try:
            return int(str(device_key).rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            return None

    def _camera_entries(self) -> List[Tuple[int, str, Dict[str, Any]]]:
        """Return canonical camera indices independent of Linux video numbering."""
        from utils.recording_device_roles import nedf_camera_entries

        return nedf_camera_entries(self.devices, CAMERA_DEVICE_TYPES)

    @staticmethod
    def _resolve_camera_image_topic(camera_id: str) -> str:
        """Map ``camera_N`` to ``nmx/hal/camera/camera_N/rgbd`` (see ``HAL_CAMERA_TOPICS``)."""
        try:
            camera_idx = int(str(camera_id).rsplit("_", 1)[-1])
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Invalid camera_id {camera_id!r}; expected form 'camera_<index>'"
            ) from exc
        if 0 <= camera_idx < len(HAL_CAMERA_TOPICS):
            return HAL_CAMERA_TOPICS[camera_idx]
        raise ValueError(
            f"No HAL image topic for {camera_id!r}; "
            f"supported indices 0..{len(HAL_CAMERA_TOPICS) - 1}"
        )

    def collect_data(self) -> None:
        """
        Collect data from all devices in a non-blocking manner.

        Each device is polled independently:
        - High-frequency devices (e.g., cameras) update frequently
        - Low-frequency devices (e.g., robot states) update less often
        - Only new data (based on timestamp changes) is collected
        - Data is either queued for async writing or buffered for batch saving
        """
        current_time = time.time()

        # Read each device independently - non-blocking
        # This allows high-freq devices to update without waiting for low-freq ones
        for device_key in self.devices:
            # Skip non-recorded devices (e.g. ZED FPV, teleop command) BEFORE the
            # SHM read/copy so we don't waste cycles on frames we'd just drop.
            if self.devices[device_key].get('type') in IGNORED_RECORD_DEVICE_TYPES:
                continue

            # Read latest data for this device from shared memory
            # _read_device_data() is non-blocking - just reads from SHM buffer
            data = self._read_device_data(device_key)

            if data is not None:
                timestamp_ns, data_array = data

                # Only process if this is new data (timestamp changed)
                if timestamp_ns != self.devices[device_key]['last_timestamp']:

                    # Option 1: Async writing - queue data for background thread
                    if self.async_write and self.write_accepting:
                        try:
                            # Try non-blocking put first
                            device_info = self.devices[device_key]
                            write_task = {
                                'device_key': device_key,
                                'device_type': device_info['type'],
                                'device_name': device_info['name'],
                                'timestamp_ns': timestamp_ns,
                                'data_array': data_array.copy(),  # Copy to avoid race conditions
                                'device_info': device_info,
                                'collect_time': current_time
                            }
                            self.write_queue.put_nowait(write_task)
                        except queue.Full:
                            # Queue is full, try blocking put with timeout (10ms)
                            # This gives the writer thread a chance to process some data
                            try:
                                self.write_queue.put(write_task, timeout=0.01)
                            except queue.Full:
                                # Still full after timeout, drop this data point
                                logger.warning(f"Write queue full after timeout, dropping data for {device_key} "
                                               f"(queue size: {self.write_queue.qsize()}/{self.write_queue.maxsize})")

                    # Option 2: Buffer data for batch saving (original behavior)
                    else:
                        self.data_buffer[device_key].append((timestamp_ns, data_array))

                    # Update tracking
                    self.devices[device_key]['last_timestamp'] = timestamp_ns
                    self.devices[device_key]['last_update_time'] = current_time

        # Note: We do NOT use read_all_device_data() here because:
        # 1. It reads ALL devices from the SAME frame timestamp
        # 2. This would force all devices to be synchronized
        # 3. We want each device to update independently at its own frequency

    def _finalize_nedf3_episode(self, episode_dir: str) -> None:
        """
        Finalize NEDF3 episode by renaming directory.
        If target directory exists, it will be overwritten.

        Args:
            episode_dir: Directory containing NEDF3 files
        """
        try:
            import shutil

            # Finalize by stripping the "_temp" suffix, keeping the UUID name
            # (episode_<uuid4>_temp -> episode_<uuid4>). Matches the DM3/OSS layout.
            episode_name = os.path.basename(episode_dir)
            final_episode_name = (
                episode_name[: -len("_temp")]
                if episode_name.endswith("_temp")
                else episode_name
            )
            final_episode_dir = os.path.join(self.output_dir, final_episode_name)

            if os.path.exists(episode_dir):
                # Remove existing final directory if it exists
                if os.path.exists(final_episode_dir):
                    logger.warning(f"Overwriting existing episode: {final_episode_dir}")
                    shutil.rmtree(final_episode_dir)

                os.rename(episode_dir, final_episode_dir)
                self.last_finalized_episode_dir = final_episode_dir
                logger.info(f"Finalized NEDF3 episode: {final_episode_dir}")

                # Inject the top-level "device_type" key into metadata.json.
                # NEDFMetadata is a pydantic model with extra="ignore", so the key
                # cannot be passed through its constructor; patch the written file.
                self._inject_metadata_device_type(final_episode_dir)

                # Run anomaly detection on the finalized episode.
                self._detect_episode_anomalies(final_episode_dir)

            self.episode_counts += 1

        except Exception as e:
            logger.error(f"Error finalizing NEDF3 episode: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def _episode_recording_times(episode_dir: str) -> Optional[Tuple[float, float, float]]:
        """Return ``(start_unix, end_unix, duration_s)`` for an episode.

        Derived from ``data/mcap_index.json`` (per-shard ``start_time_ns`` /
        ``end_time_ns``) so the live-written values match those produced by the
        backfill tool (docs/backfill_recording_times_r1.py) exactly.
        """
        index_path = os.path.join(episode_dir, "data", "mcap_index.json")
        if not os.path.exists(index_path):
            return None
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            starts = [int(e["start_time_ns"]) for e in entries if "start_time_ns" in e]
            ends = [int(e["end_time_ns"]) for e in entries if "end_time_ns" in e]
            if not starts or not ends:
                return None
            start_unix = min(starts) / 1e9
            end_unix = max(ends) / 1e9
            return start_unix, end_unix, end_unix - start_unix
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not derive recording times from {index_path}: {e}")
            return None

    def _inject_metadata_device_type(self, episode_dir: str) -> None:
        """Populate the metadata.json ``extra`` block per the R1 NEDF3 schema.

        NEDFMetadata is a pydantic model with extra="ignore", so these keys are
        patched into the written file. Fields:
          - device_type, hardware_string, codebase_version  (existing)
          - device, user_id, user_name                       (placeholders)
          - flexiv_control_mode, flexiv_impedance            (Flexiv controller,
            constant per episode; recorded from config since the RDK exposes no
            getter for live stiffness/damping)
          - eef_wrench_calibration                            (configured value;
            whether eef_wrench was bias/gravity calibrated)
          - recording_start_unix, recording_end_unix, duration  (recording time
            span in unix seconds, derived from data/mcap_index.json)
        (``collection_mode`` stays ``teleoperation`` at the top level.)
        """
        try:
            metadata_path = os.path.join(episode_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                logger.warning(f"metadata.json not found, skip extra inject: {metadata_path}")
                return
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            device_type = getattr(self, "device_type", None) or "teleoperation_r1"
            extra = meta.get("extra")
            if not isinstance(extra, dict):
                extra = {}
            extra["device_type"] = device_type
            if self.hardware_string:
                extra["hardware_string"] = self.hardware_string
            else:
                extra.pop("hardware_string", None)
                logger.warning("hardware_string is not configured; omitting it from metadata")
            extra.setdefault("codebase_version", get_git_commit_id())
            # GUI-fillable identity fields (override any prior placeholders).
            device_identifier = getattr(self, "device_identifier", "") or ""
            extra["device"] = [{"identifier": device_identifier}]
            extra["user_id"] = getattr(self, "user_id", "") or ""
            extra["user_name"] = getattr(self, "user_name", "") or ""
            # Flexiv controller params (constant within an episode).
            if self.flexiv_control_mode:
                extra["flexiv_control_mode"] = self.flexiv_control_mode
            else:
                extra.pop("flexiv_control_mode", None)
            if self.flexiv_impedance is not None:
                extra["flexiv_impedance"] = self.flexiv_impedance
            else:
                extra.pop("flexiv_impedance", None)
            if self.eef_wrench_calibration is not None:
                calibration_value = self.eef_wrench_calibration
                if isinstance(calibration_value, bool):
                    calibration_value = str(calibration_value).lower()
                extra["eef_wrench_calibration"] = calibration_value
            else:
                extra.pop("eef_wrench_calibration", None)
            # Recording time span (unix seconds), derived from data/mcap_index.json.
            times = self._episode_recording_times(episode_dir)
            if times is not None:
                start_unix, end_unix, duration = times
                extra["recording_start_unix"] = start_unix
                extra["recording_end_unix"] = end_unix
                extra["duration"] = duration
            else:
                logger.warning(
                    "Could not derive recording times (no data/mcap_index.json); "
                    "recording_start_unix/recording_end_unix/duration omitted"
                )
            meta["extra"] = extra
            # Drop any legacy top-level copies.
            meta.pop("device_type", None)
            meta.pop("hardware_string", None)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
            logger.info(
                f"Injected extra block (device_type='{device_type}', "
                f"flexiv_control_mode, flexiv_impedance, placeholders) into {metadata_path}"
            )
        except Exception as e:
            logger.error(f"Error injecting extra fields into metadata.json: {e}")

    def _discard_nedf3_episode(self, episode_dir: str) -> None:
        """
        Discard NEDF3 episode by deleting directory.
        """
        try:
            import shutil
            if os.path.exists(episode_dir):
                shutil.rmtree(episode_dir)
                logger.info(f"Discarded NEDF3 episode: {episode_dir}")
        except Exception as e:
            logger.error(f"Error discarding NEDF3 episode: {e}")
            import traceback
            traceback.print_exc()


    def start_saving(self) -> None:
        """Start the data saving process."""
        if not self.summary_shm:
            logger.error("Not connected to summary SHM")
            return

        self.running = True


        # Start async writer if enabled
        if self.async_write:
            self.start_async_writer()
            logger.info("Async write mode enabled - data will be written continuously to disk")
        else:
            logger.info("Batch mode enabled - data will be saved at intervals")

        # Start keyboard listener for manual episode discard.
        # In GUI mode the Tk window owns start/stop/discard, so skip the
        # terminal-focused pynput listener (it would steal global key events).
        if not self.gui_control:
            self._setup_keyboard_listener()

        logger.info("Start data acquisition. Ctrl+C to exit.")
        logger.info("+" * 40)

        episode_discarded = False
        try:
            while self.running:
                if self.gui_control:
                    logger.info("Waiting for GUI to start episode (press Enter / Start)...")
                    if not self._gui_wait_for_start():
                        break
                elif self.enable_button_control:
                    logger.info("Waiting for leader robot button press to start episode...")
                    self._wait_for_episode_start_signal()
                else:
                    logger.info("Press Enter to start data acquisition.")
                    wait_enter_blocking()
                if not self.running:
                    break
                self.episode_running = True
                data_folders = []
                logger.info("Data acquisition started.")
                logger.info('🟢'*20)
                time_episode_start = time.time()

                # Async mode: Open file streams for this episode
                if self.async_write:
                    # UUID-named episodes (DM3/OSS convention): bare <uuid4>, no prefix.
                    # Write into a *_temp dir first; finalize strips the suffix.
                    episode_dir = os.path.join(
                        self.output_dir, f"{uuid.uuid4()}_temp"
                    )
                    self.open_episode_streams(episode_dir)

                # Create precise timing function for collection loop
                # In async mode, this is just the polling rate
                # In batch mode, this is the save interval
                collection_interval = 0.005 if self.async_write else self.save_interval
                wait_for_next_collect = precise_loop_timing(collection_interval)

                while True:
                    # Collect data from devices - non-blocking, independent per device
                    self.collect_data()

                    # Batch mode: save data at intervals
                    if not self.async_write:
                        if (self.frame_timestamp is not None and
                                self.last_saving_timestamp != self.frame_timestamp and
                                any(self.data_buffer.values())):

                            saved_data_folder = self.save_data()
                            if saved_data_folder is not None:
                                data_folders.append(saved_data_folder)
                                self.last_saving_timestamp = self.frame_timestamp

                            if len(data_folders) != len(set(data_folders)):
                                logger.error("Data folders have redundancy")

                    # Wait for next collection interval
                    wait_for_next_collect()

                    # Check for episode end signal (GUI / button control / keyboard)
                    episode_ended = False
                    episode_discarded = False
                    if self.gui_control:
                        if self._gui_stop_event.is_set():
                            self._gui_stop_event.clear()
                            episode_ended = True
                    elif self.enable_button_control:
                        episode_ended = self._check_episode_end_signal()
                    else:
                        episode_ended = enter_pressed_nonblocking()

                    # check for episode discard signal
                    if self.gui_control and self._gui_discard_event.is_set():
                        self._gui_discard_event.clear()
                        episode_discarded = True
                    if self._discard_keyboard_pressed:
                        episode_discarded = True
                        self._discard_keyboard_pressed = False

                    # GUI shutdown requested mid-episode -> stop after finalize.
                    if self.gui_control and not self.running:
                        episode_ended = True

                    episode_ended = episode_ended or episode_discarded

                    if episode_discarded:
                        logger.info("Episode discarded via signal")

                    if episode_ended:
                        logger.info("Data acquisition stopped by user.")
                        break

                # Batch mode: save any remaining data
                if not self.async_write:
                    if all(self.data_buffer.values()) and self.last_saving_timestamp != self.frame_timestamp:
                        saved_data_folder = self.save_data()
                        if saved_data_folder is not None and saved_data_folder not in data_folders:
                            data_folders.append(saved_data_folder)

                # Async mode: Close writer and finalize NEDF3
                if self.async_write:
                    logger.info("Waiting for async writer to finish...")
                    if not self._wait_for_write_queue():
                        failure_reason = (
                            f"Episode write did not drain cleanly; preserving {episode_dir} "
                            "as an incomplete temporary episode and stopping collection"
                        )
                        logger.error(failure_reason)
                        self._record_episode_write_failure(failure_reason)
                        self._mark_episode_incomplete(failure_reason)
                        self.running = False
                    elif self._episode_write_failed:
                        failure_reason = self._episode_write_failure_reason or "Async write failed"
                        self._mark_episode_incomplete(failure_reason)
                        self.close_episode_streams()
                        self.running = False
                    else:
                        logger.info("All data written to NEDF3")
                        self.close_episode_streams()

                        if episode_discarded:
                            logger.info("Episode was discarded, skipping finalization")
                            self._discard_nedf3_episode(episode_dir)
                        else:
                            logger.info("Finalizing NEDF3 episode...")
                            self._finalize_nedf3_episode(episode_dir)
                else:
                    logger.warning("Batch mode is not supported for NEDF3 format")

                logger.info(f"Episode completed in {time.time() - time_episode_start:.2f}s")

        except KeyboardInterrupt:
            logger.info("Data saving interrupted by user")

        finally:
            # Stop keyboard listener
            self._stop_keyboard_listener()
            self.stop()

    def stop(self) -> None:
        """Stop the data saver consumer and async writer."""
        if (
            not self.running
            and self.write_thread is None
            and not self.episode_active
            and self.summary_shm is None
        ):
            return

        logger.info("Stopping data saver consumer...")

        # Stop async writer first
        self.stop_async_writer()

        if self.running:
            super().stop()
        else:
            self.disconnect()

        logger.info("Data saver consumer stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get status of the data saver consumer."""
        status = {
            'connected': self.summary_shm is not None,
            'running': self.running,
            'devices': len(self.devices),
            'output_dir': self.output_dir,
            'save_interval': self.save_interval,
            'frame_timestamp': self.frame_timestamp,
            'async_write': self.async_write,
            'write_thread_running': self.write_thread_running,
            'queue_size': self.write_queue.qsize() if self.async_write else 0,
            'anomaly_detection_enabled': self.enable_anomaly_detection,
            'button_control_enabled': self.enable_button_control
        }

        # Add buffer information
        buffer_info = {}
        for device_key, buffer_data in self.data_buffer.items():
            buffer_info[device_key] = len(buffer_data)
        status['buffer_samples'] = buffer_info

        # Add anomaly detection statistics
        if self.enable_anomaly_detection:
            status['anomaly_stats'] = self.anomaly_stats.copy()

        return status



# --- Base NEDF3 consumer machinery (inlined from the former
#     async_data_saver_NEDF3.py, now kept in docs/ for reference). ---
_BaseDataSaverConsumer = DataSaverConsumer


# --------------------------------------------------------------------------- #
# R1 schema constants
# --------------------------------------------------------------------------- #
# camera_N role labels (device_class_id N -> role); see docs sample metadata.
R1_CAMERA_ROLE_LABELS: Tuple[str, str, str] = (
    "inhand_right",
    "inhand_left",
    "head",
)

R1_RECORD_TOPIC_LIST: Tuple[str, ...] = (
    "nmx/hal/camera/camera_0/rgbd",
    "nmx/hal/arm/right/tcp/cmd",
    "nmx/hal/arm/right/tcp/state",
    "nmx/hal/arm/right/tcp/vel",
    "nmx/hal/arm/right/joint/state",
    "nmx/hal/arm/right/joint/vel",
    "nmx/hal/arm/right/eef_wrench",
    "nmx/hal/arm/right/extra",
    "nmx/hal/hand/right/cmd",
    "nmx/hal/hand/right/state",
    "nmx/hal/hand/right/extra",
    "nmx/hal/camera/camera_1/rgbd",
    "nmx/hal/arm/left/tcp/cmd",
    "nmx/hal/arm/left/tcp/state",
    "nmx/hal/arm/left/tcp/vel",
    "nmx/hal/arm/left/joint/state",
    "nmx/hal/arm/left/joint/vel",
    "nmx/hal/arm/left/eef_wrench",
    "nmx/hal/arm/left/extra",
    "nmx/hal/hand/left/cmd",
    "nmx/hal/hand/left/state",
    "nmx/hal/hand/left/extra",
    "nmx/hal/camera/camera_2/rgbd",
    "nmx/hal/head/cmd",
    "nmx/hal/head/state",
    "nmx/hal/head/joint/state",
    "nmx/hal/head/extra",
)

R1_EXTRA_KEYS: Tuple[str, ...] = ()

def _arm_topic(side: str, kind: str) -> str:
    return f"nmx/hal/arm/{side}/{kind}"


def _hand_topic(side: str, kind: str) -> str:
    return f"nmx/hal/hand/{side}/{kind}"


def _head_topic(kind: str) -> str:
    # Head arm lives under its own HAL namespace (no "arm/" prefix).
    return f"nmx/hal/head/{kind}"


# --------------------------------------------------------------------------- #
# protobuf message builders (nmx_msg)
# --------------------------------------------------------------------------- #
def _set_pose3(pose3: Any, pose7: np.ndarray) -> None:
    """Fill a nested ``Pose3`` from ``[x,y,z,qx,qy,qz,qw]`` (quat stored w-first)."""
    x, y, z, qx, qy, qz, qw = (
        float(v) for v in np.asarray(pose7, dtype=np.float64).reshape(7)
    )
    pose3.pos.x = x
    pose3.pos.y = y
    pose3.pos.z = z
    pose3.quat.w = qw
    pose3.quat.x = qx
    pose3.quat.y = qy
    pose3.quat.z = qz


def _arm_tcp_cmd(pose7: np.ndarray) -> Any:
    from nmx_msg.Arm_pb2 import ArmTcpCmd

    msg = ArmTcpCmd()
    _set_pose3(msg.pose, pose7)
    return msg


def _arm_tcp_state(pose7: np.ndarray) -> Any:
    from nmx_msg.Arm_pb2 import ArmTcpState

    msg = ArmTcpState()
    _set_pose3(msg.pose, pose7)
    return msg


def _arm_tcp_vel(vel6: np.ndarray) -> Any:
    """Build TCP velocity (6D twist) into ``ArmTcpState.linear_vel/angular_vel``.

    Input ``[vx,vy,vz,wx,wy,wz]`` (world frame, linear m/s, angular rad/s). Pose is
    left at default since this topic carries velocity only.
    """
    from nmx_msg.Arm_pb2 import ArmTcpState

    vx, vy, vz, wx, wy, wz = (
        float(v) for v in np.asarray(vel6, dtype=np.float64).reshape(6)
    )
    msg = ArmTcpState()
    msg.linear_vel.x = vx
    msg.linear_vel.y = vy
    msg.linear_vel.z = vz
    msg.angular_vel.x = wx
    msg.angular_vel.y = wy
    msg.angular_vel.z = wz
    return msg


def _arm_joint_vel(joint_vel: np.ndarray) -> Any:
    """Joint-space velocity (``ArmState.joints[i].vel`` = ``joint0..jointN``)."""
    from nmx_msg.Arm_pb2 import ArmState

    msg = ArmState()
    for i, dq in enumerate(np.asarray(joint_vel, dtype=np.float64).reshape(-1)):
        j = msg.joints.add()
        j.name = f"joint{i}"
        j.vel = float(dq)
    return msg


def _arm_end_force(wrench6: np.ndarray) -> Any:
    """Build ``ArmEndForce`` from ``[Fx,Fy,Fz,Wx,Wy,Wz]`` (TCP frame)."""
    from nmx_msg.Arm_pb2 import ArmEndForce

    fx, fy, fz, wx, wy, wz = (
        float(v) for v in np.asarray(wrench6, dtype=np.float64).reshape(6)
    )
    msg = ArmEndForce()
    msg.force.x = fx
    msg.force.y = fy
    msg.force.z = fz
    msg.torque.x = wx
    msg.torque.y = wy
    msg.torque.z = wz
    return msg


def _hand_cmd(width_m: float) -> Any:
    """Gripper opening (m) as a single ``gripper`` ``JointCmd``."""
    from nmx_msg.Hand_pb2 import HandCmd

    msg = HandCmd()
    j = msg.joints.add()
    j.name = "gripper"
    j.pos = float(width_m)
    return msg


def _hand_state(width_m: float) -> Any:
    """Gripper opening (m) as a single ``gripper`` ``JointState``."""
    from nmx_msg.Hand_pb2 import HandState

    msg = HandState()
    j = msg.joints.add()
    j.name = "gripper"
    j.pos = float(width_m)
    return msg


def _arm_state_joints(joint_values: np.ndarray) -> Any:
    """Joint-space arm state (``ArmState.joints`` = ``joint0..jointN``)."""
    from nmx_msg.Arm_pb2 import ArmState

    msg = ArmState()
    for i, q in enumerate(np.asarray(joint_values, dtype=np.float64).reshape(-1)):
        j = msg.joints.add()
        j.name = f"joint{i}"
        j.pos = float(q)
    return msg


_IDENTITY_POSE7 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Consumer
# --------------------------------------------------------------------------- #
class R1DataSaverConsumer(_BaseDataSaverConsumer):
    """NEDF3 saver for the R1 arm/hand/head HAL schema.

    Reuses the UMI saver lifecycle; overrides only metadata generation and the
    per-frame topic mapping.
    """

    # ----- metadata / writer ------------------------------------------------ #
    def open_episode_streams(self, episode_dir: str) -> None:
        try:
            os.makedirs(episode_dir, exist_ok=True)
            self.current_episode_dir = episode_dir
            self._reset_episode_write_state()
            self.topics_config = NEDFTopicsConfig()

            # camera_info: keys camera_N (sorted by device_class_id), values role labels.
            camera_entries = self._camera_entries()
            camera_info: Dict[str, str] = {}
            for cam_idx, device_key, device_info in camera_entries:
                canonical_id = f"camera_{cam_idx}"
                if cam_idx < len(R1_CAMERA_ROLE_LABELS):
                    role_label = R1_CAMERA_ROLE_LABELS[cam_idx]
                else:
                    role_label = str(device_info.get("name", canonical_id))
                camera_info[canonical_id] = role_label
                self.camera_id_mapping[device_key] = canonical_id
                self.camera_image_topic_mapping[canonical_id] = (
                    self._resolve_camera_image_topic(canonical_id)
                )

            # main_camera: camera_0 (inhand_right) per the R1 sample; fall back.
            if "camera_0" in camera_info:
                main_camera = "camera_0"
            elif camera_info:
                main_camera = next(iter(camera_info))
            else:
                main_camera = "camera_0"

            record_topic_list = list(R1_RECORD_TOPIC_LIST)

            metadata = NEDFMetadata(
                hardware_config_name="",
                task_name=self.task_name,
                task_description=self.task_description,
                robot_vendor_model=self.robot_type,
                data_format="NEDF3",
                mcap_file_name="data.mcap",
                mcap_file_type="CHUNK_FILE",
                collection_mode="teleoperation",
                end_effector="gripper",
                end_effector_vendor_model=self.end_effector_vendor_model,
                robot_type="dual_arm",
                arm_base_to_world_transform=[],
                umi_to_tcp_transform=self.umi_to_tcp_transform,
                camera_info=camera_info,
                main_camera=main_camera,
                extra_keys=list(R1_EXTRA_KEYS),
                extra={},
                record_topic_list=record_topic_list,
            )

            metadata_path = os.path.join(episode_dir, "metadata.json")
            writer_config = NEDFWriterConfig(
                metadata_file_path=Path(metadata_path),
                metadata=metadata,
                topics_config=self.topics_config,
                max_file_size_mb=100,
                mcap_chunk_size_kb=256,
            )

            self.nedf_writer_context = NEDFFactory.get_writer(NEDFType.NEDF3, writer_config)
            self.nedf_writer = self.nedf_writer_context.__enter__()

            for device_key in self.devices:
                self.device_data_counts[device_key] = 0

            self.episode_active = True
            logger.info(f"Opened R1 NEDF3 writer in {episode_dir}")
            logger.info(f"Cameras: {list(camera_info.keys())}")
            logger.info(
                f"record_topic_list ({len(record_topic_list)}): {record_topic_list}"
            )

        except Exception as e:
            logger.error(f"Error opening R1 NEDF3 writer: {e}")
            import traceback

            traceback.print_exc()
            raise

    # ----- per-side writers ------------------------------------------------- #
    def _write_arm_hand_bundle(
        self,
        side: str,
        cmd_pose7: np.ndarray,
        state_pose7: np.ndarray,
        wrench6: np.ndarray,
        gripper_width_m: float,
        timestamp_ns: int,
        tcp_vel6: Optional[np.ndarray] = None,
        joint_state: Optional[np.ndarray] = None,
        joint_vel: Optional[np.ndarray] = None,
        gripper_cmd_width_m: Optional[float] = None,
    ) -> None:
        writer = self.nedf_writer
        if writer is None:
            return
        writer.write_message(
            _arm_topic(side, "tcp/cmd"), _arm_tcp_cmd(cmd_pose7), timestamp_ns, timestamp_ns
        )
        writer.write_message(
            _arm_topic(side, "tcp/state"),
            _arm_tcp_state(state_pose7),
            timestamp_ns,
            timestamp_ns,
        )
        if tcp_vel6 is not None:
            writer.write_message(
                _arm_topic(side, "tcp/vel"),
                _arm_tcp_vel(tcp_vel6),
                timestamp_ns,
                timestamp_ns,
            )
        if joint_state is not None:
            writer.write_message(
                _arm_topic(side, "joint/state"),
                _arm_state_joints(joint_state),
                timestamp_ns,
                timestamp_ns,
            )
        if joint_vel is not None:
            writer.write_message(
                _arm_topic(side, "joint/vel"),
                _arm_joint_vel(joint_vel),
                timestamp_ns,
                timestamp_ns,
            )
        writer.write_message(
            _arm_topic(side, "eef_wrench"),
            _arm_end_force(wrench6),
            timestamp_ns,
            timestamp_ns,
        )
        writer.write_message(
            _arm_topic(side, "extra"),
            _empty_extra_metadata_message(),
            timestamp_ns,
            timestamp_ns,
        )
        # hand/cmd = the COMMANDED opening width when the robot publishes a distinct
        # commanded-width tail (v3, flat[86]/[87]); falls back to mirroring the
        # measured width when that signal is absent (teleop/older layouts).
        cmd_width = (
            float(gripper_cmd_width_m)
            if gripper_cmd_width_m is not None
            else float(gripper_width_m)
        )
        writer.write_message(
            _hand_topic(side, "cmd"), _hand_cmd(cmd_width), timestamp_ns, timestamp_ns
        )
        writer.write_message(
            _hand_topic(side, "state"),
            _hand_state(gripper_width_m),
            timestamp_ns,
            timestamp_ns,
        )
        writer.write_message(
            _hand_topic(side, "extra"),
            _empty_extra_metadata_message(),
            timestamp_ns,
            timestamp_ns,
        )

    def _write_head_bundle(
        self,
        joints: np.ndarray,
        cmd_pose7: np.ndarray,
        state_pose7: np.ndarray,
        timestamp_ns: int,
    ) -> None:
        writer = self.nedf_writer
        if writer is None:
            return
        writer.write_message(
            _head_topic("cmd"),
            _arm_tcp_cmd(cmd_pose7),
            timestamp_ns,
            timestamp_ns,
        )
        writer.write_message(
            _head_topic("state"),
            _arm_tcp_state(state_pose7),
            timestamp_ns,
            timestamp_ns,
        )
        writer.write_message(
            _head_topic("joint/state"),
            _arm_state_joints(joints),
            timestamp_ns,
            timestamp_ns,
        )
        writer.write_message(
            _head_topic("extra"),
            _empty_extra_metadata_message(),
            timestamp_ns,
            timestamp_ns,
        )

    def _write_control_shm(
        self, device_key: str, data_array: np.ndarray, timestamp_ns: int
    ) -> bool:
        """Write executed dual-arm controls or policy chunks into canonical extra topics."""
        writer = self.nedf_writer
        if writer is None:
            return False
        values = np.asarray(data_array, dtype=np.float64)
        if values.shape == (16,) and device_key.endswith("_control"):
            for side, offset in (("left", 0), ("right", 8)):
                writer.write_message(
                    _arm_topic(side, "tcp/cmd"),
                    _arm_tcp_cmd(values[offset : offset + 7]),
                    timestamp_ns,
                    timestamp_ns,
                )
                writer.write_message(
                    _hand_topic(side, "cmd"),
                    _hand_cmd(float(values[offset + 7])),
                    timestamp_ns,
                    timestamp_ns,
                )
            return True
        if values.ndim == 2 and values.shape[1] == 16:
            payload = {
                "policy_action_chunk": {
                    "source": device_key,
                    "shape": list(values.shape),
                    "values": values.tolist(),
                }
            }
            writer.write_message(
                _arm_topic("left", "extra"),
                _extra_metadata_message(payload),
                timestamp_ns,
                timestamp_ns,
            )
            return True
        logger.warning(f"Unsupported ControlSHM shape {values.shape} for {device_key}")
        return False

    # ----- async write worker ---------------------------------------------- #
    def _async_write_worker(self) -> None:
        logger.info("R1 async write worker thread started")

        while True:
            write_task = None
            try:
                write_task = self._get_next_write_task()
                if write_task is None:
                    break

                device_key = write_task["device_key"]
                device_type = write_task["device_type"]
                timestamp_ns = write_task["timestamp_ns"]
                data_array = write_task["data_array"]

                if (
                    not self.episode_active
                    or self.nedf_writer is None
                    or self.topics_config is None
                ):
                    logger.warning(
                        f"Episode not active or writer not open for {device_key}, dropping data"
                    )
                    self.write_queue.task_done()
                    continue

                if device_type in CAMERA_DEVICE_TYPES:
                    camera_id = self.camera_id_mapping.get(device_key)
                    if camera_id is None:
                        logger.warning(f"Camera ID not found for {device_key}")
                        self.write_queue.task_done()
                        continue

                    if self.image_size is not None and _camera_role_from_id(camera_id) != "head":
                        target_width, target_height = self.image_size
                        data_array = cv2.resize(
                            data_array,
                            (target_width, target_height),
                            interpolation=cv2.INTER_LINEAR,
                        )

                    # Per-role stored geometry: head -> untouched (native frame);
                    # wrist -> aspect crop + resize up to the head pixel grid.
                    data_array = _process_camera_image(data_array, camera_id)

                    if len(data_array.shape) == 3 and data_array.shape[2] == 3:
                        data_array = cv2.cvtColor(data_array, cv2.COLOR_RGB2BGR)

                    _, jpg_bytes = cv2.imencode(
                        ".jpg", data_array, [cv2.IMWRITE_JPEG_QUALITY, 95]
                    )
                    rows = int(data_array.shape[0])
                    cols = int(data_array.shape[1])
                    channels = int(data_array.shape[2]) if len(data_array.shape) == 3 else 1
                    rgbd_msg = _build_rgbd_from_jpeg(jpg_bytes.tobytes(), rows, cols, channels)

                    rgbd_topic = self.camera_image_topic_mapping.get(camera_id)
                    if not rgbd_topic:
                        rgbd_topic = self._resolve_camera_image_topic(camera_id)
                    self.nedf_writer.write_message(
                        rgbd_topic, rgbd_msg, timestamp_ns, timestamp_ns
                    )

                elif "RizonRobot" in device_type:
                    # Frame layout (R1 [88]; degrades for [86]/[74]/[48]/[44]):
                    #   [0:7]   L q / [22:29] R q                    -> joint/state
                    #   [7:14]  L dq / [29:36] R dq                   -> joint/vel
                    #   [14:21] L measured tcp [x,y,z,qw,qx,qy,qz]   -> tcp/state
                    #   [36:43] R measured tcp                        -> tcp/state
                    #   [44]/[46] L/R gripper width_m (state)         -> hand/*/state
                    #   [48:55] L commanded tcp / [55:61] L wrench    -> tcp/cmd, eef_wrench
                    #   [61:68] R commanded tcp / [68:74] R wrench
                    #   [74:80] L tcp_vel / [80:86] R tcp_vel         -> tcp/vel
                    #   [86]/[87] L/R gripper commanded width_m       -> hand/*/cmd
                    flat = np.asarray(data_array, dtype=np.float64).reshape(-1)
                    if flat.size < RizonDualArmStateLayout.BASE_SIZE:
                        logger.warning(
                            f"{device_type} {device_key}: expected >=44 values, got {flat.size}"
                        )
                        self.write_queue.task_done()
                        continue

                    left_state_pose = _flexiv_tcp_to_pose7(
                        flat[RizonDualArmStateLayout.LEFT_MEASURED_TCP]
                    )
                    right_state_pose = _flexiv_tcp_to_pose7(
                        flat[RizonDualArmStateLayout.RIGHT_MEASURED_TCP]
                    )
                    left_joint_state = flat[
                        RizonDualArmStateLayout.LEFT_JOINT_POSITION
                    ]
                    right_joint_state = flat[
                        RizonDualArmStateLayout.RIGHT_JOINT_POSITION
                    ]
                    left_joint_vel = flat[RizonDualArmStateLayout.LEFT_JOINT_VELOCITY]
                    right_joint_vel = flat[RizonDualArmStateLayout.RIGHT_JOINT_VELOCITY]

                    if flat.size >= RizonDualArmStateLayout.GRIPPER_STATE_SIZE:
                        l_width = float(flat[RizonDualArmStateLayout.LEFT_GRIPPER_WIDTH_M])
                        r_width = float(flat[RizonDualArmStateLayout.RIGHT_GRIPPER_WIDTH_M])
                    else:
                        # Teleop-shaped SHM: only normalized gripper value present.
                        l_width = float(flat[RizonDualArmStateLayout.LEFT_GRIPPER_VALUE])
                        r_width = float(flat[RizonDualArmStateLayout.RIGHT_GRIPPER_VALUE])

                    if flat.size >= RizonDualArmStateLayout.ARM_DETAIL_SIZE:
                        left_cmd_pose = _flexiv_tcp_to_pose7(
                            flat[RizonDualArmStateLayout.LEFT_COMMAND_TCP]
                        )
                        left_wrench = flat[RizonDualArmStateLayout.LEFT_EXTERNAL_WRENCH]
                        right_cmd_pose = _flexiv_tcp_to_pose7(
                            flat[RizonDualArmStateLayout.RIGHT_COMMAND_TCP]
                        )
                        right_wrench = flat[RizonDualArmStateLayout.RIGHT_EXTERNAL_WRENCH]
                    else:
                        # No command/wrench tail: reuse measured pose, zero wrench.
                        left_cmd_pose = left_state_pose
                        right_cmd_pose = right_state_pose
                        left_wrench = np.zeros(6, dtype=np.float64)
                        right_wrench = np.zeros(6, dtype=np.float64)

                    if flat.size >= RizonDualArmStateLayout.TCP_VELOCITY_SIZE:
                        left_tcp_vel = flat[RizonDualArmStateLayout.LEFT_TCP_VELOCITY]
                        right_tcp_vel = flat[RizonDualArmStateLayout.RIGHT_TCP_VELOCITY]
                    else:
                        left_tcp_vel = None
                        right_tcp_vel = None

                    if flat.size >= RizonDualArmStateLayout.COMMAND_GRIPPER_SIZE:
                        l_cmd_width = float(
                            flat[RizonDualArmStateLayout.LEFT_COMMAND_GRIPPER_WIDTH_M]
                        )
                        r_cmd_width = float(
                            flat[RizonDualArmStateLayout.RIGHT_COMMAND_GRIPPER_WIDTH_M]
                        )
                    else:
                        l_cmd_width = None
                        r_cmd_width = None

                    self._write_arm_hand_bundle(
                        "left",
                        left_cmd_pose,
                        left_state_pose,
                        left_wrench,
                        l_width,
                        timestamp_ns,
                        tcp_vel6=left_tcp_vel,
                        joint_state=left_joint_state,
                        joint_vel=left_joint_vel,
                        gripper_cmd_width_m=l_cmd_width,
                    )
                    self._write_arm_hand_bundle(
                        "right",
                        right_cmd_pose,
                        right_state_pose,
                        right_wrench,
                        r_width,
                        timestamp_ns,
                        tcp_vel6=right_tcp_vel,
                        joint_state=right_joint_state,
                        joint_vel=right_joint_vel,
                        gripper_cmd_width_m=r_cmd_width,
                    )

                elif device_type == "ControlSHM":
                    if not self._write_control_shm(device_key, data_array, timestamp_ns):
                        self.write_queue.task_done()
                        continue

                elif device_type == "YamHeadRobot":
                    # Frame layout (R1 [20]; degrades for [13]/[6]):
                    #   [0:6]   measured joints                 -> joint/state
                    #   [6:13]  commanded FK [x,y,z,qx,qy,qz,qw]-> tcp/cmd
                    #   [13:20] measured FK                     -> tcp/state
                    flat = np.asarray(data_array, dtype=np.float64).reshape(-1)
                    if flat.size >= 20:
                        joints = flat[0:6]
                        cmd_pose = flat[6:13]
                        state_pose = flat[13:20]
                    elif flat.size >= 13:
                        joints = flat[0:6]
                        cmd_pose = flat[6:13]
                        state_pose = flat[6:13]
                    elif flat.size >= 6:
                        joints = flat[0:6]
                        cmd_pose = _IDENTITY_POSE7
                        state_pose = _IDENTITY_POSE7
                    else:
                        logger.warning(
                            f"{device_type} {device_key}: expected >=6 values, got {flat.size}"
                        )
                        self.write_queue.task_done()
                        continue
                    self._write_head_bundle(joints, cmd_pose, state_pose, timestamp_ns)

                else:
                    logger.warning(f"Unknown device type: {device_type} for {device_key}")
                    self.write_queue.task_done()
                    continue

                if device_key not in self.device_data_counts:
                    self.device_data_counts[device_key] = 0
                self.device_data_counts[device_key] += 1

                self.write_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                if write_task is not None:
                    self.write_queue.task_done()
                self._record_episode_write_failure(f"R1 async write worker error: {e}")
                logger.error(f"Error in R1 async write worker: {e}")
                import traceback

                traceback.print_exc()

        self.write_thread_running = False
        logger.info("R1 async write worker thread stopped")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="R1 Data Saver Consumer - Save Device Data to NEDF3 (arm/hand/head schema)"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Optional Hydra config YAML with runtime.signal_files (required if button control is on)")
    parser.add_argument("--summary-shm", "-s", default="device_summary_data",
                        help="Summary shared memory name (default: device_summary_data)")
    parser.add_argument("--output-dir", "-o", default="saved_data",
                        help="Output directory for saved data (default: saved_data)")
    parser.add_argument("--interval", "-i", type=float, default=0.001,
                        help="Save interval in seconds (default: 0.001)")
    parser.add_argument("--start-idx", "-st", type=int, default=0,
                        help="Start index of the episode")
    parser.add_argument("--image-size", "-is", type=str, default=None,
                        help="Target image size 'width,height' (e.g. '400,300'); default original")
    parser.add_argument("--async-write", "-a", action="store_true", default=True,
                        help="Use async write mode (default: True)")
    parser.add_argument("--queue-size", "-q", type=int, default=5000,
                        help="Async write queue size (default: 5000)")
    parser.add_argument(
        "--writer-shutdown-timeout",
        type=float,
        default=30.0,
        help="Seconds allowed to drain queued writes during shutdown (default: 30)",
    )
    parser.add_argument("--robot-type", "-r", type=str, default="R1-Flexiv-Rizon4",
                        help="NEDF metadata robot_vendor_model (default: R1-Flexiv-Rizon4)")
    parser.add_argument("--task-name", "-t", type=str, default="data_collection",
                        help="Task name for NEDF metadata (default: data_collection)")
    parser.add_argument("--task-description", type=str, default="",
                        help="Task description for NEDF metadata (default: empty)")
    parser.add_argument("--device-class-identifier", type=str, default="",
                        help="metadata.json extra.device[0].identifier (default: empty)")
    parser.add_argument("--user-id", type=str, default="",
                        help="metadata.json extra.user_id of the operator (default: empty)")
    parser.add_argument("--user-name", type=str, default="",
                        help="metadata.json extra.user_name of the operator (default: empty)")
    parser.add_argument("--gui", action="store_true",
                        help="Launch the Tkinter GUI (fill metadata, Enter to start/stop, replay/delete episodes)")
    parser.add_argument(
        "--end-effector-vendor-model", "-e", type=str, default="lktech_gripper",
        help="NEDF3 metadata end_effector_vendor_model (default: lktech_gripper)",
    )
    parser.add_argument(
        "--device-type", "-d", type=str, default="teleoperation_r1",
        help="Top-level metadata.json device_type (default: teleoperation_r1)",
    )
    parser.add_argument("--enable-anomaly-detection", action="store_true", default=True,
                        help="Enable episode anomaly detection (default: True)")
    parser.add_argument("--disable-anomaly-detection", action="store_true",
                        help="Disable episode anomaly detection")
    parser.add_argument("--anomaly-config", type=str, default=None,
                        help="Path to anomaly detection config file")
    parser.add_argument("--enable-button-control", action="store_true", default=True,
                        help="Enable button-controlled episode management (default: True)")
    parser.add_argument("--disable-button-control", action="store_true",
                        help="Disable button-controlled episode management (use keyboard Enter)")
    parser.add_argument("--status", action="store_true",
                        help="Show consumer status and exit")

    args = parser.parse_args()

    signal_files: Dict[str, str] = {}
    runtime_metadata: Dict[str, Any] = {}
    if args.config:
        composed_config = load_composed_config(args.config)
        runtime_cfg = composed_config.get("runtime", {})
        if isinstance(runtime_cfg, dict):
            raw_signal_files = runtime_cfg.get("signal_files", {})
            if isinstance(raw_signal_files, dict):
                signal_files = dict(raw_signal_files)
        runtime_metadata = extract_r1_runtime_metadata(composed_config)
    else:
        logger.warning(
            "No --config provided; hardware_string, Flexiv mode/impedance, and "
            "wrench calibration metadata will be omitted"
        )

    enable_anomaly_detection = (
        args.enable_anomaly_detection and not args.disable_anomaly_detection
    )
    enable_button_control = (
        args.enable_button_control and not args.disable_button_control
    )
    if args.gui:
        # The GUI owns episode start/stop (Enter / buttons), so the leader-robot
        # button-signal control is mutually exclusive with it.
        enable_button_control = False
    if enable_button_control and not args.config:
        logger.warning(
            "Button control needs runtime.signal_files from --config; disabling it. "
            "Use Enter / GUI for episodes, or pass --config with episode_signal_pattern."
        )
        enable_button_control = False

    image_size = None
    if args.image_size:
        try:
            width, height = map(int, args.image_size.split(","))
            image_size = (width, height)
            logger.info(f"Target image size: {image_size} (width, height)")
        except ValueError:
            logger.error(
                f"Invalid image size format: {args.image_size}. Expected 'width,height'"
            )
            return

    consumer = R1DataSaverConsumer(
        summary_shm_name=args.summary_shm,
        output_dir=args.output_dir,
        save_interval=args.interval,
        start_idx=args.start_idx,
        image_size=image_size,
        async_write=True,
        queue_size=args.queue_size,
        writer_shutdown_timeout_s=args.writer_shutdown_timeout,
        robot_type=args.robot_type,
        task_name=args.task_name,
        task_description=args.task_description,
        device_identifier=args.device_identifier,
        user_id=args.user_id,
        user_name=args.user_name,
        end_effector_vendor_model=args.end_effector_vendor_model,
        umi_to_tcp_transform=None,
        device_type=args.device_type,
        **runtime_metadata,
        enable_anomaly_detection=enable_anomaly_detection,
        anomaly_config_path=args.anomaly_config,
        enable_button_control=enable_button_control,
        signal_files=signal_files,
    )

    if args.status:
        status = consumer.get_status()
        logger.info("R1 Data Saver Consumer Status:")
        for key in ("connected", "running", "devices", "output_dir", "save_interval"):
            if key in status:
                logger.info(f"  {key}: {status[key]}")
        return

    logger.info("R1 Data Saver Consumer - NEDF3 arm/hand/head schema")
    logger.info("=====================================================")
    logger.info(f"Summary SHM: {args.summary_shm}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Robot vendor/model: {args.robot_type}")
    logger.info(f"End-effector vendor/model: {args.end_effector_vendor_model}")
    logger.info(f"Task name: {args.task_name}")
    logger.info(f"Anomaly detection: {enable_anomaly_detection}")
    logger.info(f"Button control: {enable_button_control}")
    logger.info("")

    if not consumer.connect():
        logger.error(
            "Failed to connect to summary SHM. Make sure the Device Manager is running."
        )
        return

    if args.gui:
        consumer.gui_control = True
        try:
            import importlib

            run_gui_app = importlib.import_module(
                "incubator.r1.data_saver_gui"
            ).run_gui_app
        except ImportError as exc:
            raise RuntimeError("Failed to import the R1 data saver GUI") from exc
        try:
            run_gui_app(consumer)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            consumer.request_shutdown()
            consumer.stop()
        return

    try:
        consumer.start_saving()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        consumer.stop()


if __name__ == "__main__":
    main()
