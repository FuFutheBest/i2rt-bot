"""
NEDF3 (HAL) → LeRobot v2.1 converter (dual-arm; optional ego pose in state/action).

Tracker pose: ``nmx/hal/umi/{left,right,ego}/pose`` (``UmiPose``: xyz + quat wxyz).
Logged samples are **tracker** poses in the Vive world frame; when
``metadata.json`` provides ``umi_to_tcp_transform`` (≥16 floats), or the top-level
``umi_to_tcp_transform`` in the config YAML when metadata is empty, the converter maps each sample to **TCP**
in world as ``T_W^{tcp} = T_W^{tracker} @ inv(M)`` with ``M`` the stored
tracker-to-TCP matrix (same layout as NEDF3 metadata).
Cameras: ``nmx/hal/camera/{id}/rgbd``. Episode rows use timestamps from
``metadata.json`` ``main_camera`` (device id); that stream must be exported and
non-empty or conversion raises. TCP pose is **interpolated** onto that timeline
(linear position + quaternion slerp);
vision gripper stays **nearest-neighbor** to ``ts`` (already camera-aligned).
Detector line angles are converted to **physical gripper width (m)** using the
packaged ``visual_angle_to_width.json`` calibration (same pipeline as
``nmx_processor`` ``width_geometry``). Vision gripper detection rotates wrist
images per arm, resizes to ROI mask size, uses detector default HSV, and
interpolates short detection gaps (``gripper.short_gap_interp_max_frames``,
default 10). Set ``gripper.module_root`` to the directory that contains
``detector.py``, ``gripper.width_calibration_json`` to the calibration file,
and prepare ``roi_mask_*.png``.
TCP quats stored as ``wxyz`` for LeRobot readers.

Usage: ``python scripts/my_nedf3_to_lerobot.py --config configs/nedf3_to_lerobot.yaml``

All runtime parameters are read from the YAML; required sections and keys match
``configs/nedf3_to_lerobot.yaml`` (see ``_run_from_config``). Schema constants
stay in this module.

Author: Yichen Xie (revised for NEDF3 HAL)
"""

import io
import json
import logging
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import av
import cv2
import numpy as np
import scipy.interpolate as si
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm.auto import tqdm

_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

logger = logging.getLogger(__name__)


TCP_DIM = 7

ARM_FEATURE_SUFFIXES = (
    "x", "y", "z",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "gripper",
)
EGO_ACTION_FEATURE_SUFFIXES = (
    "x", "y", "z",
    "quat_w", "quat_x", "quat_y", "quat_z",
)

DUAL_ARM_FEATURE_NAMES: List[str] = [
    f"{arm}_{suffix}"
    for arm in ("left", "right")
    for suffix in ARM_FEATURE_SUFFIXES
]


NEDF3_POSE_TOPIC_BY_ARM: Dict[str, str] = {
    "left":  "nmx/hal/umi/left/pose",
    "right": "nmx/hal/umi/right/pose",
    "ego":   "nmx/hal/umi/ego/pose",
}

# Back-compat alias for call sites still named "TCP topic".
NEDF3_TCP_TOPIC_BY_ARM = NEDF3_POSE_TOPIC_BY_ARM

NEDF3_CAMERA_TOPIC_FMT = "nmx/hal/camera/{camera_id}/rgbd"

# Vision gripper width on left/right only; ego has TCP pose only (same as processed_nedf3).
GRIPPER_ARM_NAMES = frozenset({"left", "right"})


def _arm_has_gripper(arm_name: str) -> bool:
    return arm_name in GRIPPER_ARM_NAMES


# Matches gripper_process ``DEFAULT_CFG_JSON`` for consistent offline vs ingest behaviour.
OFFLINE_PROCESS_DEFAULT_CFG_JSON: str = """{
    "erode_px": 0,
    "min_contour_area": 1.5,
    "roi_padding": 40
}"""


def _cfg_section(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    section = cfg.get(key)
    if not isinstance(section, dict):
        raise ValueError(f"Config must contain a '{key}:' mapping (object).")
    return section


def _cfg_require_keys(
    section: Dict[str, Any], keys: Tuple[str, ...], location: str
) -> None:
    missing = [k for k in keys if k not in section]
    if missing:
        raise ValueError(
            f"Config section {location!r} is missing required key(s): {missing}"
        )


def _build_modality_mapping(feature_names: List[str]) -> Dict[str, Dict[str, int]]:
    return {
        feature_name: {"start": index, "end": index + 1}
        for index, feature_name in enumerate(feature_names)
    }


def get_dual_arm_modality_config(
    state_feature_names: Optional[List[str]] = None,
    action_feature_names: Optional[List[str]] = None,
) -> Dict:
    """Modality config for a dual-arm setup."""
    resolved_state_names = state_feature_names or DUAL_ARM_FEATURE_NAMES
    resolved_action_names = action_feature_names or DUAL_ARM_FEATURE_NAMES
    return {
        "state": _build_modality_mapping(resolved_state_names),
        "action": _build_modality_mapping(resolved_action_names),
        "video": {},
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }


def _extract_log_time_ns(msg: Any) -> int:
    """Best-effort log timestamp (ns) extraction across SDK wrappers/payloads."""
    for container in (getattr(msg, "message", None), msg):
        if container is None:
            continue
        for attr in ("log_time", "timestamp_ns", "timestamp", "publish_time"):
            value = getattr(container, attr, None)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    continue
    return 0


def _extract_decoded_message(msg: Any) -> Any:
    """Return decoded payload from an SDK message wrapper."""
    return getattr(msg, "decoded_message", msg)


def _iter_topic_messages(reader: Any, topic: str) -> Iterator[Any]:
    """Iterate decoded messages for a topic across SDK variants."""
    if hasattr(reader, "iter_decoded_messages"):
        yield from reader.iter_decoded_messages(topic)
        return
    if hasattr(reader, "iter_decoded_message"):
        yield from reader.iter_decoded_message(topic=topic)
        return
    if hasattr(reader, "get_lowdim"):
        yield from reader.get_lowdim(topic)
        return
    raise AttributeError(
        "NEDF reader has no iter_decoded_messages / iter_decoded_message / get_lowdim"
    )


def _extract_image_buffer(decoded_msg: Any) -> Optional[np.ndarray]:
    """Locate the JPG byte payload inside a decoded NEDF3 message and return a numpy view."""
    for attr in (None, "rgb", "image"):
        container = decoded_msg if attr is None else getattr(decoded_msg, attr, None)
        if container is None:
            continue
        data = getattr(container, "data", None)
        if data:
            return np.frombuffer(data, dtype=np.uint8)
    return None


def _decode_rgb_buffer(rgb_buffer: np.ndarray) -> Optional[np.ndarray]:
    """Decode a JPG (or H.264/H.265 fallback) buffer to an RGB numpy image."""
    rgb_img = cv2.imdecode(rgb_buffer, cv2.IMREAD_COLOR)
    if rgb_img is not None:
        return cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    raw_bytes = rgb_buffer.tobytes()
    for fmt in (None, "h264", "hevc"):
        try:
            open_kwargs: Dict[str, Any] = {"mode": "r"}
            if fmt is not None:
                open_kwargs["format"] = fmt
            with av.open(io.BytesIO(raw_bytes), **open_kwargs) as container:
                for frame in container.decode(video=0):
                    return frame.reformat(format="rgb24").to_ndarray()
        except Exception:
            continue
    return None


def _reorder_tcp_xyzw_to_wxyz(pose_xyzw: np.ndarray) -> List[float]:
    """Convert a NEDF3 HAL TCP sample ``[x,y,z,qx,qy,qz,qw]`` to ``[x,y,z,qw,qx,qy,qz]``."""
    p = np.asarray(pose_xyzw, dtype=np.float64).reshape(7)
    return [
        float(p[0]), float(p[1]), float(p[2]),
        float(p[6]),  # qw
        float(p[3]),  # qx
        float(p[4]),  # qy
        float(p[5]),  # qz
    ]


def _row_major_4x4_to_mat(flat16: np.ndarray) -> np.ndarray:
    """Reshape 16 row-major floats to a 4×4 matrix."""
    v = np.asarray(flat16, dtype=np.float64).reshape(16)
    return v.reshape(4, 4)


def _pose_vec_wxyz_to_T4(pose_wxyz: Union[List[float], np.ndarray]) -> np.ndarray:
    """Build 4×4 ``T_world_body`` from ``[x,y,z,qw,qx,qy,qz]`` (body origin in world, scipy quat)."""
    p = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    x, y, z, qw, qx, qy, qz = p
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = (x, y, z)
    return T


def _T4_to_pose_vec_wxyz(T: np.ndarray) -> List[float]:
    """Inverse of :func:`_pose_vec_wxyz_to_T4` (returns ``[x,y,z,qw,qx,qy,qz]``)."""
    Rm = np.asarray(T, dtype=np.float64)[:3, :3]
    qx, qy, qz, qw = R.from_matrix(Rm).as_quat()
    x, y, z = T[:3, 3]
    return [float(x), float(y), float(z), float(qw), float(qx), float(qy), float(qz)]


def _read_umi_to_tcp_transform_flat(reader: Any, metadata_path: Path) -> List[float]:
    """Return ``umi_to_tcp_transform`` floats from the reader or ``metadata.json``."""
    meta = getattr(reader, "metadata", None)
    raw = getattr(meta, "umi_to_tcp_transform", []) if meta is not None else []
    if raw:
        try:
            return [float(x) for x in raw]
        except (TypeError, ValueError):
            pass
    try:
        with open(metadata_path, "r", encoding="utf-8") as mf:
            d = json.load(mf)
        raw2 = d.get("umi_to_tcp_transform", [])
        if raw2:
            return [float(x) for x in raw2]
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return []


def _matrix4x4_to_row_major_flat(matrix: Any, *, label: str) -> List[float]:
    """Flatten one 4×4 (nested rows or 16 scalars) to row-major floats."""
    if not isinstance(matrix, (list, tuple)):
        raise ValueError(f"{label}: expected a 4×4 matrix (list), got {type(matrix)!r}")
    if len(matrix) == 0:
        raise ValueError(f"{label}: empty matrix")
    if isinstance(matrix[0], (list, tuple)):
        rows = []
        for row_idx, row in enumerate(matrix):
            row_arr = np.asarray(row, dtype=np.float64).reshape(-1)
            if row_arr.size != 4:
                raise ValueError(
                    f"{label}: row {row_idx} must have 4 values, got {row_arr.size}"
                )
            rows.append(row_arr)
        return [float(x) for x in np.stack(rows, axis=0).reshape(-1)]
    flat = [float(x) for x in matrix]
    if len(flat) != 16:
        raise ValueError(f"{label}: expected 16 row-major floats, got {len(flat)}")
    return flat


def _parse_umi_to_tcp_transform_from_config(raw: Any) -> List[float]:
    """Parse top-level YAML ``umi_to_tcp_transform`` (flat 16/32 or list of 4×4)."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)) and not raw:
        return []
    if isinstance(raw, (list, tuple)) and isinstance(raw[0], (int, float)):
        return [float(x) for x in raw]

    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            "umi_to_tcp_transform must be a flat float list or a list of 4×4 matrices."
        )

    out: List[float] = []
    for idx, item in enumerate(raw):
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            out.extend(_matrix4x4_to_row_major_flat(item, label=f"umi_to_tcp_transform[{idx}]"))
            continue
        raise ValueError(
            f"umi_to_tcp_transform[{idx}] must be a 4×4 matrix (list of rows), got {type(item)!r}"
        )
    return out


def _resolve_umi_to_tcp_transform_flat(
    reader: Any,
    metadata_path: Path,
    *,
    backup_flat: Optional[List[float]] = None,
) -> Tuple[List[float], str]:
    """Return row-major tracker→TCP floats and a source tag (metadata / yaml_backup / none)."""
    from_meta = _read_umi_to_tcp_transform_flat(reader, metadata_path)
    if len(from_meta) >= 16:
        return from_meta, "metadata"

    backup = list(backup_flat or [])
    if len(backup) >= 16:
        logger.info(
            "umi_to_tcp_transform: metadata has %d float(s); using YAML backup (%d float(s))",
            len(from_meta),
            len(backup),
        )
        return backup, "yaml_backup"

    if from_meta:
        logger.warning(
            "umi_to_tcp_transform: metadata has only %d float(s) (<16); TCP remap skipped",
            len(from_meta),
        )
    return [], "none"


def _apply_umi_tracker_to_tcp_poses(
    pose_data_by_arm: Dict[str, Dict[int, List[float]]],
    umi_flat: List[float],
) -> None:
    """
    In-place: convert stored **tracker** poses to **TCP** in the same world frame.

    ``umi_flat`` is NEDF3 ``umi_to_tcp_transform``: row-major 4×4 ``M`` (tracker→TCP).
    For each tracker sample ``T_W^{tracker}``, set ``T_W^{tcp} = T_W^{tracker} @ M^{-1}``.
    32 floats → left ``M`` then right ``M``; 16 floats → same ``M`` for both arms.
    ``ego`` streams are left unchanged.
    """
    u = np.asarray(umi_flat, dtype=np.float64).ravel()
    if u.size < 16:
        return
    M_left = _row_major_4x4_to_mat(u[:16])
    M_right = _row_major_4x4_to_mat(u[16:32]) if u.size >= 32 else M_left.copy()
    inv_left = np.linalg.inv(M_left)
    inv_right = np.linalg.inv(M_right)

    for arm, inv_m in (("left", inv_left), ("right", inv_right)):
        arm_dict = pose_data_by_arm.get(arm)
        if not arm_dict:
            continue
        for ts in list(arm_dict.keys()):
            Tw_tr = _pose_vec_wxyz_to_T4(arm_dict[ts])
            Tw_tcp = Tw_tr @ inv_m
            arm_dict[ts] = _T4_to_pose_vec_wxyz(Tw_tcp)


def _resolve_demo_root(demo_path: Path) -> Path:
    if (demo_path / "metadata.json").exists():
        return demo_path

    candidate_dirs = sorted(
        [d for d in demo_path.glob("*") if d.is_dir() and not d.name.startswith(".")]
    )
    for candidate in candidate_dirs:
        if (candidate / "metadata.json").exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find metadata.json in demo directory: {demo_path}"
    )


def _list_demo_dirs(data_dir: Path) -> List[Path]:
    if (data_dir / "metadata.json").exists():
        return [data_dir]
    return sorted([d for d in data_dir.glob("*") if d.is_dir()], key=lambda x: x.name)


def _normalize_metadata_data_format(value: Any) -> str:
    """Normalize ``metadata.json`` ``data_format`` (e.g. ``NEDFType.NEDF3`` → ``NEDF3``)."""
    if value is None:
        return ""
    s = str(value).strip()
    if "." in s:
        return s.rsplit(".", 1)[-1]
    return s


def _load_nedf3_metadata(demo_path: Path) -> Tuple[Path, Dict]:
    demo_dir = _resolve_demo_root(demo_path)
    metadata_path = demo_dir / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata_dict = json.load(f)

    fmt = _normalize_metadata_data_format(metadata_dict.get("data_format"))
    if fmt != "NEDF3":
        raise ValueError(
            f"Expected NEDF3 format, but got {metadata_dict.get('data_format')!r} "
            f"(normalized {fmt!r})"
        )
    return metadata_path, metadata_dict


def _infer_arm_names(topic_keys: List[str], include_ego: bool = False) -> List[str]:
    """Detect arms from explicit NEDF3 HAL TCP topics (no UMI fallback)."""
    topic_set = set(topic_keys)
    arms: List[str] = []
    for name in ("left", "right"):
        if NEDF3_TCP_TOPIC_BY_ARM[name] in topic_set:
            arms.append(name)
    if include_ego and NEDF3_TCP_TOPIC_BY_ARM["ego"] in topic_set:
        arms.append("ego")
    return arms


def _infer_camera_id(
    camera_info: Dict[str, str], required_keywords: List[str]
) -> Optional[str]:
    for cam_id, cam_alias in camera_info.items():
        alias = str(cam_alias).lower()
        if all(keyword in alias for keyword in required_keywords):
            return cam_id
    return None


def _resolve_camera_ids(
    metadata_dict: Dict,
    wrist_cam_1: Optional[str],
    wrist_cam_2: Optional[str],
    global_cam: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    camera_info = metadata_dict.get("camera_info") or {}
    normalized_camera_info = {
        str(cam_id): str(cam_alias) for cam_id, cam_alias in camera_info.items()
    }

    left_wrist = wrist_cam_1 or _infer_camera_id(
        normalized_camera_info, ["left", "wrist"]
    )
    right_wrist = wrist_cam_2 or _infer_camera_id(
        normalized_camera_info, ["right", "wrist"]
    )
    global_view = (
        global_cam
        or _infer_camera_id(normalized_camera_info, ["head"])
        or _infer_camera_id(normalized_camera_info, ["global"])
    )

    return left_wrist, right_wrist, global_view


def _build_feature_names(arm_names: List[str], dim: int) -> List[str]:
    if (
        tuple(arm_names) == ("left", "right")
        and dim == len(DUAL_ARM_FEATURE_NAMES)
    ):
        return DUAL_ARM_FEATURE_NAMES
    if len(arm_names) > 0:
        feature_names = []
        for arm_name in arm_names:
            feature_suffixes = (
                EGO_ACTION_FEATURE_SUFFIXES
                if not _arm_has_gripper(arm_name)
                else ARM_FEATURE_SUFFIXES
            )
            feature_names.extend(
                [f"{arm_name}_{feature_suffix}" for feature_suffix in feature_suffixes]
            )
        if len(feature_names) == dim:
            return feature_names
    return [f"motor_{idx}" for idx in range(dim)]


def _resize_image_to_target(img: np.ndarray, image_size: tuple) -> np.ndarray:
    target_h, target_w = image_size
    h, w = img.shape[:2]

    if target_h == target_w:
        if h > w:
            img = img[h // 2 - w // 2: h // 2 + w // 2, :]
        else:
            img = img[:, w // 2 - h // 2: w // 2 + h // 2]
        return cv2.resize(img, (target_w, target_h))

    if h * target_w == w * target_h:
        return cv2.resize(img, (target_w, target_h))

    raise ValueError(
        "Only support target image size with 1:1 ratio or the same aspect ratio as "
        f"the source image. Got target size {image_size} and source size {(h, w)}."
    )


def _coerce_image_size(value: Any, field_name: str) -> Tuple[int, int]:
    """Normalise a YAML image-size value to ``(height, width)`` (int or [h, w])."""
    if value is None:
        raise ValueError(f"{field_name} is required in the config YAML.")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{field_name} must be positive, got {value}.")
        return (value, value)
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            size = int(value[0])
            if size <= 0:
                raise ValueError(f"{field_name} must be positive, got {size}.")
            return (size, size)
        if len(value) == 2:
            height, width = int(value[0]), int(value[1])
            if height <= 0 or width <= 0:
                raise ValueError(
                    f"{field_name} height and width must be positive, "
                    f"got {(height, width)}."
                )
            return (height, width)
    raise ValueError(
        f"{field_name} expects an int, a 1-element list, or a 2-element "
        f"[height, width] list; got {value!r}."
    )


def _extract_pose7_xyzw(decoded: Any) -> Optional[np.ndarray]:
    """Return ``[x,y,z,qx,qy,qz,qw]`` from ``UmiPose`` or legacy lowdim ``data``."""
    pose = getattr(decoded, "pose", None)
    if pose is not None and hasattr(pose, "pos") and hasattr(pose, "quat"):
        pos = pose.pos
        quat = pose.quat
        return np.array(
            [
                float(pos.x),
                float(pos.y),
                float(pos.z),
                float(quat.x),
                float(quat.y),
                float(quat.z),
                float(quat.w),
            ],
            dtype=np.float64,
        )
    raw = getattr(decoded, "data", None)
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size < 7:
        return None
    return arr[:7]


def _fill_tcp_values(
    reader: Any, tcp_topic: str, arm_pose_dict: Dict[int, List[float]]
) -> None:
    """Populate ``{timestamp_ns: [x,y,z,qw,qx,qy,qz]}`` from a NEDF3 HAL pose topic."""
    for msg in _iter_topic_messages(reader, tcp_topic):
        ts = _extract_log_time_ns(msg)
        decoded = _extract_decoded_message(msg)
        arr = _extract_pose7_xyzw(decoded)
        if arr is None:
            continue
        arm_pose_dict.setdefault(ts, _reorder_tcp_xyzw_to_wxyz(arr))


def _load_arm_streams(
    reader: Any,
    topic_keys: List[str],
    demo_dir: Path,
    state_arm_names: List[str],
    action_arm_names: List[str],
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load TCP poses per arm; gripper dicts stay empty until vision pass. Raises if required arm has no TCP."""
    pose_data_by_arm: Dict[str, Dict[int, List[float]]] = {}
    pose_timestamps_by_arm: Dict[str, np.ndarray] = {}
    state_data_by_arm: Dict[str, Dict[int, List[float]]] = {}
    state_timestamps_by_arm: Dict[str, np.ndarray] = {}

    topic_set = set(topic_keys)
    required_arms = set(state_arm_names) | set(action_arm_names)

    for arm in dict.fromkeys(state_arm_names + action_arm_names):
        tcp_topic = NEDF3_TCP_TOPIC_BY_ARM.get(arm)

        pose_dict: Dict[int, List[float]] = {}
        if tcp_topic and tcp_topic in topic_set:
            _fill_tcp_values(reader, tcp_topic, pose_dict)

        if not pose_dict and arm in required_arms:
            raise ValueError(
                f"TCP topic '{tcp_topic}' for required arm '{arm}' contains "
                f"no samples in {demo_dir}."
            )

        pose_data_by_arm[arm] = pose_dict
        pose_timestamps_by_arm[arm] = np.array(sorted(pose_dict.keys()))
        state_data_by_arm[arm] = {}
        state_timestamps_by_arm[arm] = np.array([])

    return (
        pose_data_by_arm,
        pose_timestamps_by_arm,
        state_data_by_arm,
        state_timestamps_by_arm,
    )


def _load_camera_streams(
    reader: Any,
    topic_keys: List[str],
    demo_dir: Path,
    camera_keys: List[Tuple[str, str, tuple]],
) -> Tuple[Dict, Dict]:
    camera_data_by_key: Dict[str, Dict[int, np.ndarray]] = {}
    camera_timestamps_by_key: Dict[str, np.ndarray] = {}

    topic_set = set(topic_keys)
    for cam_key, cam_id, _ in camera_keys:
        topic = NEDF3_CAMERA_TOPIC_FMT.format(camera_id=cam_id)
        images_by_ts: Dict[int, np.ndarray] = {}

        if topic in topic_set:
            for msg in _iter_topic_messages(reader, topic):
                ts = _extract_log_time_ns(msg)
                decoded = _extract_decoded_message(msg)
                jpg_buffer = _extract_image_buffer(decoded)
                if jpg_buffer is None:
                    continue
                rgb_img = _decode_rgb_buffer(jpg_buffer)
                if rgb_img is not None:
                    images_by_ts[ts] = rgb_img
        else:
            logger.warning(
                f"Camera topic '{topic}' missing in {demo_dir}. "
                f"Camera '{cam_id}' will be filled with zeros."
            )

        camera_data_by_key[cam_key] = images_by_ts
        camera_timestamps_by_key[cam_key] = (
            np.array(sorted(images_by_ts.keys())) if images_by_ts else np.array([])
        )

    return camera_data_by_key, camera_timestamps_by_key


def _shift_camera_streams_by_latency(
    camera_data_by_key: Dict[str, Dict[int, np.ndarray]],
    camera_keys: List[Tuple[str, str, tuple]],
    latency_wrist_cam_1_s: float,
    latency_wrist_cam_2_s: float,
    latency_global_cam_s: float,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, np.ndarray]]:
    """Subtract per-camera latency from HAL log timestamps (nanoseconds).

    Each stream's dict keys become ``ts - round(latency_s * 1e9)`` so gripper
    and alignment use time-aligned video. Latencies map to LeRobot keys:
    ``wrist_image_1`` / ``wrist_image_2`` / ``observation.images.image``.
    """
    latency_s_by_key: Dict[str, float] = {
        "observation.images.wrist_image_1": latency_wrist_cam_1_s,
        "observation.images.wrist_image_2": latency_wrist_cam_2_s,
        "observation.images.image": latency_global_cam_s,
    }
    new_data: Dict[str, Dict[int, np.ndarray]] = {}
    new_ts: Dict[str, np.ndarray] = {}
    for cam_key, _, _ in camera_keys:
        latency_s = float(latency_s_by_key.get(cam_key, 0.0))
        offset_ns = int(round(latency_s * 1e9))
        old = camera_data_by_key.get(cam_key, {})
        if offset_ns == 0:
            new_data[cam_key] = dict(old)
        else:
            shifted: Dict[int, np.ndarray] = {}
            for ts, img in old.items():
                shifted[int(ts) - offset_ns] = img
            new_data[cam_key] = shifted
        keys = new_data[cam_key].keys()
        new_ts[cam_key] = (
            np.array(sorted(keys), dtype=np.int64) if keys else np.array([], dtype=np.int64)
        )
    return new_data, new_ts


def _observation_key_for_camera_id(
    camera_keys: List[Tuple[str, str, tuple]],
    camera_id: Optional[str],
) -> Optional[str]:
    """Map NEDF3 ``camera_info`` / ``main_camera`` device id to LeRobot feature key."""
    if not camera_id or not str(camera_id).strip():
        return None
    cid = str(camera_id).strip()
    for cam_key, cam_dev_id, _ in camera_keys:
        if str(cam_dev_id) == cid:
            return cam_key
    return None


def _alignment_timestamps_ns(
    camera_timestamps_by_key: Dict[str, np.ndarray],
    camera_keys: List[Tuple[str, str, tuple]],
    metadata_dict: Dict[str, Any],
    demo_dir: Path,
) -> np.ndarray:
    """Timestamps (nanoseconds) for ``_align_episode_streams`` from ``metadata.main_camera``.

    Raises:
        ValueError: If ``main_camera`` is missing, does not match an exported camera,
            or that camera has no decoded frames in this demo.
    """
    main_raw = metadata_dict.get("main_camera")
    main_id = str(main_raw).strip() if main_raw is not None else ""
    if not main_id:
        raise ValueError(
            f"{demo_dir / 'metadata.json'}: required key 'main_camera' (non-empty NEDF3 "
            "camera device id, e.g. 'camera_2') is missing or blank."
        )

    align_key = _observation_key_for_camera_id(camera_keys, main_id)
    if align_key is None:
        exported = [d for _, d, _ in camera_keys]
        raise ValueError(
            f"{demo_dir}: metadata main_camera={main_id!r} does not match any exported "
            f"camera device id in this run. Exported ids: {exported}. "
            "Ensure cameras.* in the YAML enables that camera."
        )

    main_ts = camera_timestamps_by_key.get(align_key)
    if main_ts is None or main_ts.size == 0:
        raise ValueError(
            f"{demo_dir}: main_camera={main_id!r} maps to {align_key!r} but that stream "
            "has no decoded frames (missing topic, decode failure, or empty recording). "
            f"Expected HAL topic {NEDF3_CAMERA_TOPIC_FMT.format(camera_id=main_id)}."
        )

    return main_ts


# Vision gripper: stateful detector; run frames in time order (see ``_populate_gripper_from_vision``).

_GRIPPER_DETECTOR_BY_ROOT: Dict[str, Any] = {}
_VISUAL_WIDTH_CALIBRATION: Any = None
_VISUAL_WIDTH_CALIBRATION_PATH: Optional[Path] = None


def _import_gripper_detector(module_root: Path) -> Any:
    """Lazily import :class:`GripperDetector` from the package at ``module_root``.

    ``detector.py`` uses package-relative imports (``from .config import ...``), so
    we prepend **the parent of** ``module_root`` to ``sys.path`` and load
    ``<package_name>.detector`` where ``package_name == module_root.name``.

    ``module_root`` must be the resolved directory that contains ``detector.py``.
    """
    module_root = Path(module_root).expanduser().resolve()
    cache_key = str(module_root)
    cached = _GRIPPER_DETECTOR_BY_ROOT.get(cache_key)
    if cached is not None:
        return cached

    import sys
    from importlib import import_module

    det_path = module_root / "detector.py"
    if not det_path.is_file():
        raise FileNotFoundError(
            f"Gripper detector not found: expected {det_path}. "
            "Set gripper.module_root in the config YAML to the package directory."
        )

    package_name = module_root.name
    parent_dir = str(module_root.parent.resolve())
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    detector_mod = import_module(f"{package_name}.detector")
    GripperDetector = getattr(detector_mod, "GripperDetector")

    try:
        detect_mod = import_module(f"{package_name}.detect")
        if hasattr(detect_mod, "DEBUG"):
            detect_mod.DEBUG = False
    except Exception:
        logger.debug("Could not silence gripper detect.DEBUG; continuing.")

    _GRIPPER_DETECTOR_BY_ROOT[cache_key] = GripperDetector
    return GripperDetector


def _resolve_cfg_json(cfg_value: Optional[str]) -> str:
    """Accept either an inline JSON string or a path to a JSON file."""
    if not cfg_value:
        return ""
    p = Path(cfg_value)
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return cfg_value


def _default_width_calibration_json_path(config_path: Path) -> Path:
    """Default: ``nedf3_processer/configs/visual_angle_to_width.json``."""
    del config_path
    return (
        Path(__file__).resolve().parent
        / "configs"
        / "visual_angle_to_width.json"
    ).resolve()


def _load_visual_width_calibration(calibration_path: Path) -> Any:
    """Load CAD + affine calibration once per process."""
    global _VISUAL_WIDTH_CALIBRATION, _VISUAL_WIDTH_CALIBRATION_PATH
    resolved = calibration_path.expanduser().resolve()
    if (
        _VISUAL_WIDTH_CALIBRATION is not None
        and _VISUAL_WIDTH_CALIBRATION_PATH == resolved
    ):
        return _VISUAL_WIDTH_CALIBRATION

    import sys

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from visual_width_geometry import load_visual_width_calibration

    _VISUAL_WIDTH_CALIBRATION_PATH = resolved
    _VISUAL_WIDTH_CALIBRATION = load_visual_width_calibration(resolved)
    logger.info(
        "Loaded gripper width calibration %s (max_width_m=%.6f)",
        _VISUAL_WIDTH_CALIBRATION.calibration_id,
        _VISUAL_WIDTH_CALIBRATION.physical_width_max_mm / 1000.0,
    )
    return _VISUAL_WIDTH_CALIBRATION


def _marker_angle_deg_to_width_m(
    arm: str,
    angle_deg: Optional[float],
    *,
    calibration: Any,
) -> float:
    """Map detector line angle to physical width (m); invalid detection -> 0.0 m."""
    if angle_deg is None or not math.isfinite(float(angle_deg)):
        return 0.0
    import sys

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from visual_width_geometry import estimate_visual_width_from_angle_deg

    return float(
        estimate_visual_width_from_angle_deg(
            arm, float(angle_deg), calibration
        ).width_m
    )


def _cfg_json_with_default_hsv(cfg_json: str) -> str:
    """Return cfg JSON text with HSV bounds removed so detector uses code defaults."""
    try:
        cfg_dict = json.loads(cfg_json)
    except json.JSONDecodeError:
        return cfg_json
    if not isinstance(cfg_dict, dict):
        return cfg_json
    cfg_dict.pop("hsv_lower", None)
    cfg_dict.pop("hsv_upper", None)
    return json.dumps(cfg_dict, ensure_ascii=False)


def _prepare_detector_input_bgr(rgb_img: np.ndarray, arm: str) -> np.ndarray:
    """Build detector input from an RGB frame without modifying the source image.

    For ROI alignment with offline calibration:
    - ``left`` arm uses a 90° clockwise-rotated image.
    - ``right`` arm uses a 90° counterclockwise-rotated image.

    Rotation is applied **only** for vision gripper detection; LeRobot image
    export still uses the original camera orientation.
    """
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    if arm == "left":
        return cv2.rotate(bgr_img, cv2.ROTATE_90_CLOCKWISE)
    if arm == "right":
        return cv2.rotate(bgr_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr_img


def _infer_roi_mask_size(masks_dir: Union[str, Path]) -> Optional[Tuple[int, int]]:
    """Infer ROI mask size ``(height, width)`` from ``roi_mask_*.png`` files."""
    masks_path = Path(masks_dir)
    for mask_path in sorted(masks_path.glob("roi_mask_*.png")):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.size == 0:
            continue
        h, w = mask.shape[:2]
        return (int(h), int(w))
    return None


def _fill_short_detection_gaps(
    *,
    timestamps: List[int],
    raw_values: List[float],
    detected_flags: List[bool],
    short_gap_interp_max_frames: int,
) -> Tuple[List[float], Set[int], int, int]:
    """Fill short missing runs with linear interpolation, keep long runs invalid.

    Returns:
        filled_values: per-frame gripper widths (m) after interpolation.
        invalid_timestamps: timestamps that stayed invalid (long gaps / edge gaps).
        interpolated_frame_count: number of filled missing frames.
        dropped_frame_count: number of invalid frames kept as 0.0 m.
    """
    filled_values = list(raw_values)
    invalid_timestamps: Set[int] = set()
    interpolated_frame_count = 0
    dropped_frame_count = 0

    n = len(raw_values)
    i = 0
    while i < n:
        if detected_flags[i]:
            i += 1
            continue

        start = i
        while i < n and not detected_flags[i]:
            i += 1
        end = i - 1
        gap_len = end - start + 1

        has_left_anchor = start > 0 and detected_flags[start - 1]
        has_right_anchor = i < n and detected_flags[i]
        can_interpolate = (
            gap_len <= short_gap_interp_max_frames
            and has_left_anchor
            and has_right_anchor
        )

        if can_interpolate:
            left_value = float(filled_values[start - 1])
            right_value = float(raw_values[i])
            denom = float(gap_len + 1)
            for offset in range(gap_len):
                alpha = float(offset + 1) / denom
                filled_values[start + offset] = (1.0 - alpha) * left_value + alpha * right_value
                interpolated_frame_count += 1
        else:
            for idx in range(start, end + 1):
                invalid_timestamps.add(int(timestamps[idx]))
                filled_values[idx] = 0.0
                dropped_frame_count += 1

    return (
        filled_values,
        invalid_timestamps,
        interpolated_frame_count,
        dropped_frame_count,
    )


def _detect_gripper_width_per_image(
    detector: Any,
    images_by_ts: Dict[int, np.ndarray],
    *,
    arm: str,
    calibration: Any,
    short_gap_interp_max_frames: int,
    detector_input_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[int, List[float]], Set[int], int, int]:
    """Run the gripper detector per frame and emit physical width in meters.

    Camera frames decoded by :func:`_load_camera_streams` are RGB; detector
    input is built via :func:`_prepare_detector_input_bgr` (per-arm rotation,
    optional resize to ROI mask size) before
    :meth:`GripperDetector.process_for_export`.

    Short invalid runs (``<= short_gap_interp_max_frames``) are linearly
    interpolated in width (m); long runs are left at ``0.0`` m.
    """
    sorted_ts = sorted(images_by_ts.keys())
    raw_values: List[float] = []
    detected_flags: List[bool] = []
    for ts in sorted_ts:
        rgb_img = images_by_ts[ts]
        detector_bgr_img = _prepare_detector_input_bgr(rgb_img, arm)
        if detector_input_size is not None:
            target_h, target_w = detector_input_size
            if detector_bgr_img.shape[:2] != (target_h, target_w):
                src_h, src_w = detector_bgr_img.shape[:2]
                src_ratio = float(src_w) / float(src_h)
                dst_ratio = float(target_w) / float(target_h)
                if abs(src_ratio - dst_ratio) > 1e-6:
                    raise ValueError(
                        "Aspect ratio mismatch before detector resize for arm "
                        f"'{arm}': source {(src_h, src_w)} (w/h={src_ratio:.6f}) vs "
                        f"target {(target_h, target_w)} (w/h={dst_ratio:.6f})."
                    )
                detector_bgr_img = cv2.resize(
                    detector_bgr_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )
        result = detector.process_for_export(detector_bgr_img)
        width_m = _marker_angle_deg_to_width_m(
            arm, result.angle_deg, calibration=calibration
        )
        raw_values.append(float(width_m))
        detected_flags.append(result.angle_deg is not None)

    (
        filled_values,
        invalid_timestamps,
        interpolated_frame_count,
        dropped_frame_count,
    ) = _fill_short_detection_gaps(
        timestamps=[int(ts) for ts in sorted_ts],
        raw_values=raw_values,
        detected_flags=detected_flags,
        short_gap_interp_max_frames=short_gap_interp_max_frames,
    )

    widths_by_ts: Dict[int, List[float]] = {
        int(ts): [float(v)] for ts, v in zip(sorted_ts, filled_values)
    }
    return (
        widths_by_ts,
        invalid_timestamps,
        interpolated_frame_count,
        dropped_frame_count,
    )


def _populate_gripper_from_vision(
    arm_to_cam_key: Dict[str, str],
    masks_dir_by_arm: Dict[str, str],
    cfg_json_by_arm: Dict[str, str],
    camera_data_by_key: Dict[str, Dict[int, np.ndarray]],
    state_data_by_arm: Dict[str, Dict[int, List[float]]],
    state_timestamps_by_arm: Dict[str, np.ndarray],
    gripper_module_root: Path,
    width_calibration_path: Path,
    short_gap_interp_max_frames: int,
    demo_dir: Path,
) -> None:
    """Run vision-based gripper detection per arm and write the results into
    ``state_data_by_arm`` / ``state_timestamps_by_arm``.

    Values are **physical gripper width in meters** (``aligned_corrected_gripper_width``
    semantics), not raw marker line angles.

    The resulting dicts have the same shape as the ones that used to come from
    the NEDF3 ``*/gripper_state`` topic, so the alignment loop in
    :func:`_align_episode_streams` and the per-frame alignment in
    :func:`_align_camera_image` work unchanged.
    """
    if not arm_to_cam_key:
        return
    GripperDetector = _import_gripper_detector(gripper_module_root)
    calibration = _load_visual_width_calibration(width_calibration_path)

    for arm, cam_key in arm_to_cam_key.items():
        masks_dir = masks_dir_by_arm.get(arm)
        if not masks_dir:
            # Skip arms that didn't get a masks dir; validation upstream decides
            # whether that's an error or simply "leave the gripper slot at 0.0".
            continue
        if cam_key not in camera_data_by_key:
            logger.warning(
                "Vision gripper for arm '%s': camera key '%s' not loaded in %s.",
                arm, cam_key, demo_dir,
            )
            continue
        images_by_ts = camera_data_by_key[cam_key]
        if not images_by_ts:
            logger.warning(
                "Vision gripper for arm '%s': no images on %s in %s.",
                arm, cam_key, demo_dir,
            )
            continue

        cfg_json = _resolve_cfg_json(cfg_json_by_arm.get(arm))
        if not cfg_json:
            cfg_json = OFFLINE_PROCESS_DEFAULT_CFG_JSON
        detector_cfg_json = _cfg_json_with_default_hsv(cfg_json)
        logger.info(
            "Vision gripper for arm '%s': forcing detector HSV to code defaults; "
            "ignoring hsv_lower/hsv_upper from cfg_json.",
            arm,
        )
        detector = GripperDetector(masks_dir, detector_cfg_json)
        detector.reset()
        detector_input_size = _infer_roi_mask_size(masks_dir)
        if detector_input_size is None:
            logger.warning(
                "Vision gripper for arm '%s': failed to infer ROI mask size in %s; "
                "detector input will keep source frame size.",
                arm,
                masks_dir,
            )
        else:
            logger.info(
                "Vision gripper for arm '%s': detector input size set to ROI mask size %s.",
                arm,
                detector_input_size,
            )
        (
            widths,
            invalid_timestamps,
            interpolated_frame_count,
            dropped_frame_count,
        ) = _detect_gripper_width_per_image(
            detector,
            images_by_ts,
            arm=arm,
            calibration=calibration,
            short_gap_interp_max_frames=short_gap_interp_max_frames,
            detector_input_size=detector_input_size,
        )

        state_data_by_arm[arm] = widths
        state_timestamps_by_arm[arm] = np.array(sorted(widths.keys()))
        if invalid_timestamps:
            logger.warning(
                "Vision gripper for arm '%s': %d frame(s) with long detection gaps "
                "set to 0.0 m (timestamps not dropped from episode).",
                arm,
                len(invalid_timestamps),
            )
        valid_widths = [row[0] for row in widths.values() if row]
        if valid_widths:
            logger.info(
                "Vision gripper for arm '%s': %d width samples from %s in %s "
                "(m range %.4f .. %.4f)",
                arm,
                len(widths),
                cam_key,
                demo_dir,
                min(valid_widths),
                max(valid_widths),
            )
            if interpolated_frame_count > 0:
                logger.info(
                    "Vision gripper for arm '%s': interpolated %d short-gap frame(s) "
                    "(max gap=%d).",
                    arm,
                    interpolated_frame_count,
                    short_gap_interp_max_frames,
                )
            if dropped_frame_count > 0:
                logger.warning(
                    "Vision gripper for arm '%s': %d frame(s) remain invalid after "
                    "interpolation (written as 0.0 m).",
                    arm,
                    dropped_frame_count,
                )
        else:
            logger.info(
                "Vision gripper for arm '%s': %d width samples from %s in %s",
                arm, len(widths), cam_key, demo_dir,
            )


def _find_nearest_timestamp(target_ts, available_timestamps: np.ndarray):
    idx = np.searchsorted(available_timestamps, target_ts)
    if idx == 0:
        return available_timestamps[0]
    if idx == len(available_timestamps):
        return available_timestamps[-1]
    if abs(available_timestamps[idx] - target_ts) < abs(
        available_timestamps[idx - 1] - target_ts
    ):
        return available_timestamps[idx]
    return available_timestamps[idx - 1]


def _interp1d_axis0(t: np.ndarray, x: np.ndarray) -> si.interp1d:
    """Interpolate one-dimensional samples along time."""
    return si.interp1d(
        t,
        x,
        axis=0,
        bounds_error=False,
        fill_value=(x[0], x[-1]),
    )


class TcpPoseInterpolator:
    """Interpolate 7D TCP samples ``[x,y,z,qw,qx,qy,qz]`` in log time.

    Position is linearly interpolated; orientation uses ``Rotation.Slerp`` on
    scipy quaternions ``(qx, qy, qz, qw)``.
    """

    def __init__(self, t_ns: np.ndarray, poses_wxyz: np.ndarray) -> None:
        self._t = np.asarray(t_ns, dtype=np.float64).reshape(-1)
        self._poses = np.asarray(poses_wxyz, dtype=np.float64)
        if self._t.size != self._poses.shape[0]:
            raise ValueError(
                f"TcpPoseInterpolator: len(t)={self._t.size} != poses rows={self._poses.shape[0]}"
            )
        if self._t.size == 0:
            self._t0 = 0.0
            self._mode = "empty"
            return
        # Relative times (seconds or ns) so interp1d/Slerp inputs stay well-scaled vs wall ns.
        self._t0 = float(self._t[0])
        self._t = self._t - self._t0
        if self._t.size == 1:
            self._mode = "constant"
            self._const = self._poses[0].copy()
            return
        pos = self._poses[:, :3]
        qw = self._poses[:, 3]
        qx = self._poses[:, 4]
        qy = self._poses[:, 5]
        qz = self._poses[:, 6]
        quat_xyzw = np.column_stack([qx, qy, qz, qw])
        rot = R.from_quat(quat_xyzw)
        self._pos_i = _interp1d_axis0(self._t, pos)
        self._rot_s = Slerp(self._t, rot)
        self._mode = "interp"

    def sample(self, t_query_ns: Union[int, float, np.number]) -> List[float]:
        """Interpolate pose at ``t_query_ns`` (same time unit as constructor ``t_ns``)."""
        if self._mode == "empty":
            return [0.0] * 7
        if self._mode == "constant":
            return [float(x) for x in self._const]
        tq = float(t_query_ns) - self._t0
        t_min = float(self._t[0])
        t_max = float(self._t[-1])
        tq = min(max(tq, t_min), t_max)
        pos = self._pos_i(np.array([tq], dtype=np.float64))[0]
        rot_q = self._rot_s(np.array([tq], dtype=np.float64))
        qx, qy, qz, qw = rot_q.as_quat()[0]
        return [
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(qw),
            float(qx),
            float(qy),
            float(qz),
        ]


def _build_tcp_pose_interpolators_by_arm(
    pose_data_by_arm: Dict[str, Dict[int, List[float]]],
    pose_timestamps_by_arm: Dict[str, np.ndarray],
) -> Dict[str, TcpPoseInterpolator]:
    """One :class:`TcpPoseInterpolator` per arm with TCP samples (log timestamps in ns)."""
    out: Dict[str, TcpPoseInterpolator] = {}
    for arm_name, ts_arr in pose_timestamps_by_arm.items():
        ts_np = np.asarray(ts_arr, dtype=np.float64).reshape(-1)
        if ts_np.size == 0:
            continue
        pdict = pose_data_by_arm.get(arm_name, {})
        rows: List[np.ndarray] = []
        times: List[float] = []
        for t in ts_np:
            vec = pdict.get(int(t))
            if vec is None:
                continue
            rows.append(np.asarray(vec, dtype=np.float64).reshape(7))
            times.append(float(t))
        if len(rows) < 1:
            continue
        t_stack = np.asarray(times, dtype=np.float64)
        pose_stack = np.stack(rows, axis=0)
        out[arm_name] = TcpPoseInterpolator(t_stack, pose_stack)
    return out


def _append_aligned_pose_and_gripper(
    out: List[float],
    ts,
    arm_name: str,
    pose_data_by_arm: Dict,
    pose_timestamps_by_arm: Dict,
    state_data_by_arm: Dict,
    state_timestamps_by_arm: Dict,
    pose_interp_by_arm: Dict[str, TcpPoseInterpolator],
) -> None:
    """Append TCP pose (interpolated) + gripper width in m (nearest timestamp) at ``ts``."""
    interp = pose_interp_by_arm.get(arm_name)
    if interp is None:
        return

    pose = interp.sample(ts)

    if not _arm_has_gripper(arm_name):
        out.extend(pose)
        return

    state_timestamps = state_timestamps_by_arm.get(arm_name, np.array([]))
    if state_timestamps.size > 0:
        nearest_state_ts = _find_nearest_timestamp(ts, state_timestamps)
        gripper = state_data_by_arm.get(arm_name, {}).get(nearest_state_ts, [0.0])
    else:
        gripper = [0.0]

    out.extend(pose + gripper)


def _align_camera_image(
    ts,
    cam_key: str,
    target_size: tuple,
    camera_data_by_key: Dict,
    camera_timestamps_by_key: Dict,
) -> np.ndarray:
    """Nearest camera frame to ``ts``, resized; zeros if missing (gripper runs in pre-pass, not here)."""
    camera_timestamps = camera_timestamps_by_key[cam_key]
    if camera_timestamps.size > 0:
        nearest_cam_ts = _find_nearest_timestamp(ts, camera_timestamps)
        img = camera_data_by_key[cam_key].get(nearest_cam_ts)
        if img is not None:
            return _resize_image_to_target(img, target_size)
    return np.zeros((*target_size, 3), dtype=np.uint8)


def _align_episode_streams(
    timestamps: np.ndarray,
    state_arm_names: List[str],
    action_arm_names: List[str],
    camera_keys: List[Tuple[str, str, tuple]],
    pose_data_by_arm: Dict,
    pose_timestamps_by_arm: Dict,
    state_data_by_arm: Dict,
    state_timestamps_by_arm: Dict,
    camera_data_by_key: Dict,
    camera_timestamps_by_key: Dict,
    pose_interp_by_arm: Dict[str, TcpPoseInterpolator],
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Align state, action, and images to ``timestamps`` (``metadata.main_camera`` timeline, ns)."""
    aligned_states: List[List[float]] = []
    aligned_actions: List[List[float]] = []
    aligned_images: Dict[str, List[np.ndarray]] = {cam_key: [] for cam_key, _, _ in camera_keys}
    camera_size_by_key = {
        cam_key: cam_image_size for cam_key, _, cam_image_size in camera_keys
    }

    for ts in timestamps:
        state: List[float] = []
        action: List[float] = []

        for arm_name in state_arm_names:
            _append_aligned_pose_and_gripper(
                out=state,
                ts=ts,
                arm_name=arm_name,
                pose_data_by_arm=pose_data_by_arm,
                pose_timestamps_by_arm=pose_timestamps_by_arm,
                state_data_by_arm=state_data_by_arm,
                state_timestamps_by_arm=state_timestamps_by_arm,
                pose_interp_by_arm=pose_interp_by_arm,
            )

        aligned_states.append(state)
        for action_arm_name in action_arm_names:
            _append_aligned_pose_and_gripper(
                out=action,
                ts=ts,
                arm_name=action_arm_name,
                pose_data_by_arm=pose_data_by_arm,
                pose_timestamps_by_arm=pose_timestamps_by_arm,
                state_data_by_arm=state_data_by_arm,
                state_timestamps_by_arm=state_timestamps_by_arm,
                pose_interp_by_arm=pose_interp_by_arm,
            )

        aligned_actions.append(action)
        for cam_key, _, _ in camera_keys:
            aligned_images[cam_key].append(
                _align_camera_image(
                    ts=ts,
                    cam_key=cam_key,
                    target_size=camera_size_by_key[cam_key],
                    camera_data_by_key=camera_data_by_key,
                    camera_timestamps_by_key=camera_timestamps_by_key,
                )
            )

    return (
        np.array(aligned_states, dtype=np.float32),
        np.array(aligned_actions, dtype=np.float32),
        aligned_images,
    )


def _add_episode_frames(
    dataset,
    demo_dir: Path,
    final_task_desc: str,
    timestamps: np.ndarray,
    aligned_states: np.ndarray,
    aligned_actions: np.ndarray,
    aligned_images: Dict,
    camera_keys: List[Tuple[str, str, tuple]],
) -> None:
    num_frames = len(timestamps) - 1  # action is the next state
    if num_frames <= 0:
        logger.warning(f"Not enough frames in {demo_dir}, skipping...")
        return

    for i in range(num_frames):
        frame = {
            "observation.state": aligned_states[i],
            "action": aligned_actions[i + 1],
        }
        for cam_key, _, _ in camera_keys:
            frame[cam_key] = aligned_images[cam_key][i]
        dataset.add_frame(frame, task=final_task_desc)

    dataset.save_episode()


def _build_base_features(
    state_dim: int,
    action_dim: int,
    state_feature_names: List[str],
    action_feature_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": {"motors": state_feature_names},
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": {"motors": action_feature_names},
        },
    }


def _add_video_feature(
    features: Dict[str, Dict[str, Any]],
    camera_keys: List[Tuple[str, str, tuple]],
    feature_key: str,
    camera_id: Optional[str],
    image_size: tuple,
) -> None:
    if not camera_id:
        return
    features[feature_key] = {
        "dtype": "video",
        "shape": (*image_size, 3),
        "names": ["height", "width", "channel"],
    }
    camera_keys.append((feature_key, camera_id, image_size))


def _build_camera_features_and_keys(
    features: Dict[str, Dict[str, Any]],
    metadata_dict: Dict,
    wrist_cam_1: Optional[str],
    wrist_cam_2: Optional[str],
    global_cam: Optional[str],
    wrist_image_size: tuple,
    global_image_size: tuple,
) -> Tuple[
    List[Tuple[str, str, tuple]],
    Optional[str],
    Optional[str],
    Optional[str],
]:
    wrist_cam_1, wrist_cam_2, global_cam = _resolve_camera_ids(
        metadata_dict=metadata_dict,
        wrist_cam_1=wrist_cam_1,
        wrist_cam_2=wrist_cam_2,
        global_cam=global_cam,
    )
    camera_keys: List[Tuple[str, str, tuple]] = []
    _add_video_feature(
        features,
        camera_keys,
        "observation.images.wrist_image_1",
        wrist_cam_1,
        wrist_image_size,
    )
    _add_video_feature(
        features,
        camera_keys,
        "observation.images.image",
        global_cam,
        global_image_size,
    )
    _add_video_feature(
        features,
        camera_keys,
        "observation.images.wrist_image_2",
        wrist_cam_2,
        wrist_image_size,
    )
    return camera_keys, wrist_cam_1, wrist_cam_2, global_cam


def _write_modality_config(
    dataset_output_dir: Path,
    modality_config: Optional[Dict],
    camera_keys: List[Tuple[str, str, tuple]],
    state_feature_names: List[str],
    action_feature_names: List[str],
) -> None:
    if modality_config is None:
        modality_config = get_dual_arm_modality_config(
            state_feature_names=state_feature_names,
            action_feature_names=action_feature_names,
        )

    for cam_key, _, _ in camera_keys:
        short_name = cam_key.split(".")[-1]
        modality_config["video"][short_name] = {"original_key": cam_key}

    with open(dataset_output_dir / "meta" / "modality.json", "w") as f:
        json.dump(modality_config, f, indent=4)


def _split_demos_to_subsets(
    demos: List[Path], num_subsets: int, seed: int
) -> List[List[Path]]:
    random.seed(seed)
    shuffled = demos.copy()
    random.shuffle(shuffled)
    return [shuffled[i::num_subsets] for i in range(num_subsets)]


def convert_nedf3_to_lerobot(
    latency_wrist_cam_1: float,
    latency_wrist_cam_2: float,
    latency_global_cam: float,
    data_dir: Union[str, Path],
    dataset_name: str,
    *,
    output_dir: Union[str, Path],
    wrist_cam_1: Optional[str],
    wrist_cam_2: Optional[str],
    global_cam: Optional[str],
    task_desc: Optional[str],
    num_demos: int,
    fps: int,
    ee_dim: int,
    omit_ego_state_action: bool,
    image_size: tuple,
    global_image_size: tuple,
    gripper_cam_by_arm: Dict[str, str],
    gripper_masks_by_arm: Dict[str, str],
    gripper_cfg_by_arm: Dict[str, str],
    gripper_module_root: Path,
    gripper_width_calibration_path: Path,
    gripper_short_gap_interp_max_frames: int,
    backup_umi_to_tcp_transform: List[float],
    modality_config: Optional[Dict] = None,
    pre_filtered_demos: Optional[List[Path]] = None,
) -> None:
    """Convert NEDF3 HAL demos to LeRobot (dual-arm; vision gripper needs masks per arm)."""
    av.logging.set_level(av.logging.ERROR)

    # Lazy imports to avoid heavy startup cost on `--help`.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from nmx_nedf_api import NEDFFactory, NEDFReaderConfig

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    if pre_filtered_demos is not None:
        demos = pre_filtered_demos
    else:
        demos = _list_demo_dirs(data_dir)
    if num_demos > 0:
        demos = demos[:num_demos]

    if not demos:
        raise ValueError(f"No demo directories found in {data_dir}")

    # Determine robot configuration from the first demo.
    metadata_path, metadata_dict = _load_nedf3_metadata(demos[0])
    reader = NEDFFactory.get_reader(
        NEDFReaderConfig(metadata_file_path=str(metadata_path))
    )
    topics = list(reader.get_raw_topics_timestamps().keys())

    arm_names = _infer_arm_names(topics, include_ego=False)
    if not arm_names:
        raise ValueError(
            f"No NEDF3 HAL TCP topics found in {metadata_path}. "
            f"Expected at least one of {list(NEDF3_TCP_TOPIC_BY_ARM.values())[:2]}."
        )

    num_arms = len(arm_names)
    robot_type = reader.metadata.robot_type or (
        "dual_arm" if num_arms == 2 else "single_arm"
    )
    ego_topic_present = (
        not omit_ego_state_action
        and num_arms == 2
        and NEDF3_TCP_TOPIC_BY_ARM["ego"] in set(topics)
    )
    include_ego_pose = ego_topic_present

    state_arm_names = arm_names + (["ego"] if include_ego_pose else [])
    state_dim = sum(
        TCP_DIM + (ee_dim if _arm_has_gripper(arm_name) else 0)
        for arm_name in state_arm_names
    )
    action_arm_names = arm_names + (["ego"] if include_ego_pose else [])
    action_dim = sum(
        TCP_DIM + (ee_dim if _arm_has_gripper(arm_name) else 0)
        for arm_name in action_arm_names
    )

    dataset_output_dir = output_dir / dataset_name
    if dataset_output_dir.exists():
        shutil.rmtree(dataset_output_dir)

    state_feature_names = _build_feature_names(state_arm_names, state_dim)
    action_feature_names = _build_feature_names(action_arm_names, action_dim)

    wrist_image_size = image_size
    resolved_global_image_size = global_image_size
    features = _build_base_features(
        state_dim=state_dim,
        action_dim=action_dim,
        state_feature_names=state_feature_names,
        action_feature_names=action_feature_names,
    )

    camera_keys, wrist_cam_1, wrist_cam_2, global_cam = _build_camera_features_and_keys(
        features=features,
        metadata_dict=metadata_dict,
        wrist_cam_1=wrist_cam_1,
        wrist_cam_2=wrist_cam_2,
        global_cam=global_cam,
        wrist_image_size=wrist_image_size,
        global_image_size=resolved_global_image_size,
    )

    # ---- Vision-based gripper config: validate & resolve from YAML ----------
    resolved_gripper_cam_by_arm: Dict[str, str] = dict(gripper_cam_by_arm)
    resolved_gripper_masks_by_arm: Dict[str, str] = dict(gripper_masks_by_arm)
    resolved_gripper_cfg_by_arm: Dict[str, str] = dict(gripper_cfg_by_arm)

    # Vision gripper on left/right only; ego is TCP-only (aligned with processed_nedf3).
    arms_needing_gripper = {
        arm
        for arm in set(state_arm_names) | set(action_arm_names)
        if _arm_has_gripper(arm)
    }
    cam_key_set = {ck for ck, _, _ in camera_keys}
    missing_masks: List[str] = []
    bad_cameras: List[Tuple[str, str]] = []
    for arm in sorted(arms_needing_gripper):
        cam_key = resolved_gripper_cam_by_arm.get(arm)
        masks_dir = resolved_gripper_masks_by_arm.get(arm)
        if not masks_dir:
            missing_masks.append(arm)
            continue
        if not cam_key:
            raise ValueError(
                f"Set gripper.{arm}.camera in the config YAML (LeRobot image key for that arm)."
            )
        if cam_key not in cam_key_set:
            bad_cameras.append((arm, cam_key))
    if missing_masks:
        raise ValueError(
            "Vision-based gripper detection is mandatory but no masks directory "
            f"is configured for arm(s): {missing_masks}. "
            f"Set gripper.<arm>.masks_dir in the config YAML for each arm."
        )
    if bad_cameras:
        raise ValueError(
            "Vision gripper cameras not enabled in this conversion: "
            + ", ".join(f"{a} -> '{ck}'" for a, ck in bad_cameras)
            + f". Available camera keys: {sorted(cam_key_set)}."
        )
    resolved_gripper_cam_by_arm = {
        arm: ck for arm, ck in resolved_gripper_cam_by_arm.items()
        if arm in arms_needing_gripper and resolved_gripper_masks_by_arm.get(arm)
    }
    logger.info(
        "Vision gripper config: arm -> (camera, masks_dir): %s",
        {
            arm: (ck, resolved_gripper_masks_by_arm.get(arm))
            for arm, ck in resolved_gripper_cam_by_arm.items()
        },
    )

    logger.info("Detected arm names: %s", arm_names)
    if omit_ego_state_action:
        logger.info(
            "dataset.omit_ego_state_action=true: ego/head TCP excluded from "
            "state/action (left/right only)."
        )
    elif include_ego_pose:
        logger.info(
            "Ego/head TCP topic present; appending ego trajectory (7D, no gripper) "
            "to state/action (%s).",
            NEDF3_TCP_TOPIC_BY_ARM["ego"],
        )
    logger.info(
        "Using camera ids - wrist_cam_1: %s, global_cam: %s, wrist_cam_2: %s",
        wrist_cam_1,
        global_cam,
        wrist_cam_2,
    )
    logger.info(
        "Using image size - wrist: %s, global: %s",
        wrist_image_size,
        resolved_global_image_size,
    )

    logger.info(
        "Camera latency correction (YAML ms -> s): wrist_cam_1=%.6f, wrist_cam_2=%.6f, global_cam=%.6f",
        latency_wrist_cam_1,
        latency_wrist_cam_2,
        latency_global_cam,
    )

    dataset = LeRobotDataset.create(
        root=dataset_output_dir,
        repo_id=dataset_name,
        fps=fps,
        features=features,
        robot_type=robot_type,
        use_videos=True,
        image_writer_threads=4,
    )

    for demo_dir in tqdm(demos, desc="Converting demos..."):
        _process_single_demo(
            demo_dir=demo_dir,
            dataset=dataset,
            camera_keys=camera_keys,
            task_desc=task_desc,
            state_arm_names=state_arm_names,
            action_arm_names=action_arm_names,
            gripper_cam_by_arm=resolved_gripper_cam_by_arm,
            gripper_masks_by_arm=resolved_gripper_masks_by_arm,
            gripper_cfg_by_arm=resolved_gripper_cfg_by_arm,
            gripper_module_root=gripper_module_root,
            gripper_width_calibration_path=gripper_width_calibration_path,
            gripper_short_gap_interp_max_frames=gripper_short_gap_interp_max_frames,
            backup_umi_to_tcp_transform=backup_umi_to_tcp_transform,
            latency_wrist_cam_1_s=latency_wrist_cam_1,
            latency_wrist_cam_2_s=latency_wrist_cam_2,
            latency_global_cam_s=latency_global_cam,
        )

    _write_modality_config(
        dataset_output_dir=dataset_output_dir,
        modality_config=modality_config,
        camera_keys=camera_keys,
        state_feature_names=state_feature_names,
        action_feature_names=action_feature_names,
    )

    logger.info(f"Conversion complete! Output saved to {dataset_output_dir}")
    logger.info(
        f"Total episodes: {dataset.num_episodes}, Total frames: {dataset.num_frames}"
    )


def _process_single_demo(
    demo_dir: Path,
    dataset,
    camera_keys: List[Tuple[str, str, tuple]],
    task_desc: Optional[str],
    state_arm_names: List[str],
    action_arm_names: List[str],
    gripper_cam_by_arm: Dict[str, str],
    gripper_masks_by_arm: Dict[str, str],
    gripper_cfg_by_arm: Dict[str, str],
    gripper_module_root: Path,
    gripper_width_calibration_path: Path,
    gripper_short_gap_interp_max_frames: int,
    backup_umi_to_tcp_transform: List[float],
    latency_wrist_cam_1_s: float,
    latency_wrist_cam_2_s: float,
    latency_global_cam_s: float,
) -> None:
    """Process a single demo directory and append its frames to ``dataset``."""
    from nmx_nedf_api import NEDFFactory, NEDFReaderConfig

    demo_dir = _resolve_demo_root(demo_dir)
    metadata_path = demo_dir / "metadata.json"

    reader = NEDFFactory.get_reader(
        NEDFReaderConfig(metadata_file_path=str(metadata_path))
    )

    final_task_desc = task_desc or reader.metadata.task_name.replace("_", " ")

    all_topics_timestamps = reader.get_raw_topics_timestamps()
    topic_names = list(all_topics_timestamps.keys())
    demo_arm_names = _infer_arm_names(topic_names, include_ego=True)
    if not demo_arm_names:
        raise ValueError(
            f"No NEDF3 HAL TCP topics found in {demo_dir}. "
            "Cannot reconstruct any required arm."
        )

    # Required state/action arms must all be present in the demo. Extra arms
    # in the demo that the LeRobot dataset does NOT consume are silently
    # ignored — that's intentional and not an error.
    missing_state_arms = [a for a in state_arm_names if a not in demo_arm_names]
    if missing_state_arms:
        raise ValueError(
            f"State arms {missing_state_arms} are required by the LeRobot "
            f"dataset but are missing from {demo_dir}. "
            f"Detected arms in demo: {sorted(demo_arm_names)}."
        )

    missing_action_arms = [a for a in action_arm_names if a not in demo_arm_names]
    if missing_action_arms:
        raise ValueError(
            f"Action arms {missing_action_arms} are required by the LeRobot "
            f"dataset but are missing from {demo_dir}. "
            f"Detected arms in demo: {sorted(demo_arm_names)}."
        )

    (
        pose_data_by_arm,
        pose_timestamps_by_arm,
        state_data_by_arm,
        state_timestamps_by_arm,
    ) = _load_arm_streams(
        reader=reader,
        topic_keys=topic_names,
        demo_dir=demo_dir,
        state_arm_names=state_arm_names,
        action_arm_names=action_arm_names,
    )

    umi_flat, umi_source = _resolve_umi_to_tcp_transform_flat(
        reader,
        metadata_path,
        backup_flat=backup_umi_to_tcp_transform,
    )
    if len(umi_flat) >= 16:
        _apply_umi_tracker_to_tcp_poses(pose_data_by_arm, umi_flat)
        logger.info(
            "%s: applied umi_to_tcp_transform (%d floats, source=%s)",
            demo_dir.name,
            len(umi_flat),
            umi_source,
        )

    camera_data_by_key, camera_timestamps_by_key = _load_camera_streams(
        reader=reader,
        topic_keys=topic_names,
        demo_dir=demo_dir,
        camera_keys=camera_keys,
    )
    camera_data_by_key, camera_timestamps_by_key = _shift_camera_streams_by_latency(
        camera_data_by_key,
        camera_keys,
        latency_wrist_cam_1_s,
        latency_wrist_cam_2_s,
        latency_global_cam_s,
    )

    # Vision-based gripper detection: runs once per camera frame, in
    # chronological order, and writes back into state_data_by_arm /
    # state_timestamps_by_arm so the existing alignment loop can consume it.
    _populate_gripper_from_vision(
        arm_to_cam_key=gripper_cam_by_arm,
        masks_dir_by_arm=gripper_masks_by_arm,
        cfg_json_by_arm=gripper_cfg_by_arm,
        camera_data_by_key=camera_data_by_key,
        state_data_by_arm=state_data_by_arm,
        state_timestamps_by_arm=state_timestamps_by_arm,
        gripper_module_root=gripper_module_root,
        width_calibration_path=gripper_width_calibration_path,
        short_gap_interp_max_frames=gripper_short_gap_interp_max_frames,
        demo_dir=demo_dir,
    )

    with open(metadata_path, "r", encoding="utf-8") as mf:
        metadata_dict_demo: Dict[str, Any] = json.load(mf)

    timestamps = _alignment_timestamps_ns(
        camera_timestamps_by_key,
        camera_keys,
        metadata_dict_demo,
        demo_dir,
    )

    pose_interp_by_arm = _build_tcp_pose_interpolators_by_arm(
        pose_data_by_arm,
        pose_timestamps_by_arm,
    )

    aligned_states, aligned_actions, aligned_images = _align_episode_streams(
        timestamps=timestamps,
        state_arm_names=state_arm_names,
        action_arm_names=action_arm_names,
        camera_keys=camera_keys,
        pose_data_by_arm=pose_data_by_arm,
        pose_timestamps_by_arm=pose_timestamps_by_arm,
        state_data_by_arm=state_data_by_arm,
        state_timestamps_by_arm=state_timestamps_by_arm,
        camera_data_by_key=camera_data_by_key,
        camera_timestamps_by_key=camera_timestamps_by_key,
        pose_interp_by_arm=pose_interp_by_arm,
    )

    _add_episode_frames(
        dataset=dataset,
        demo_dir=demo_dir,
        final_task_desc=final_task_desc,
        timestamps=timestamps,
        aligned_states=aligned_states,
        aligned_actions=aligned_actions,
        aligned_images=aligned_images,
        camera_keys=camera_keys,
    )
    del aligned_images


def _resolve_gripper_config(
    gripper_section: Dict[str, Any],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Path, Optional[str], int]:
    """Extract vision-gripper settings from the YAML ``gripper`` section (per-arm blocks)."""
    cam_by_arm: Dict[str, str] = {}
    masks_by_arm: Dict[str, str] = {}
    cfg_by_arm: Dict[str, str] = {}

    for arm in ("left", "right"):
        per_arm = gripper_section.get(arm)
        if not per_arm:
            continue
        cam = per_arm.get("camera")
        if cam:
            cam_by_arm[arm] = str(cam)
        masks_dir = per_arm.get("masks_dir")
        if masks_dir:
            masks_by_arm[arm] = str(masks_dir)
        cfg_value = per_arm.get("cfg_json")
        if cfg_value:
            cfg_by_arm[arm] = str(cfg_value)

    raw_root = gripper_section.get("module_root")
    if raw_root is None or not str(raw_root).strip():
        raise ValueError(
            "gripper.module_root must be set to a non-empty path: the directory "
            "that contains detector.py (package root). There is no implicit default."
        )
    module_root = Path(str(raw_root).strip())

    raw_calibration = gripper_section.get("width_calibration_json")
    width_calibration_json = (
        str(raw_calibration).strip() if raw_calibration is not None else None
    )
    if width_calibration_json == "":
        width_calibration_json = None

    raw_short_gap = gripper_section.get("short_gap_interp_max_frames", 10)
    try:
        short_gap_interp_max_frames = int(raw_short_gap)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "gripper.short_gap_interp_max_frames must be an integer >= 0."
        ) from exc
    if short_gap_interp_max_frames < 0:
        raise ValueError("gripper.short_gap_interp_max_frames must be >= 0.")

    return (
        cam_by_arm,
        masks_by_arm,
        cfg_by_arm,
        module_root,
        width_calibration_json,
        short_gap_interp_max_frames,
    )


def _strip_leading_repo_dir_name(relative: Path, repo_root: Path) -> Path:
    """
    Drop the first path segment when it equals ``repo_root.name``.

    Some YAML examples prefix paths with the repository folder (e.g.
    ``SuperInference/data/...``). Resolving from the repo root must not nest
    that name twice when the checkout directory is already ``SuperInference``.
    """
    parts = relative.parts
    if parts and parts[0] == repo_root.name:
        return Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return relative


def _resolve_project_path(raw: Union[str, Path], config_path: Path) -> Path:
    """Resolve YAML paths: try SuperInference repo root, then ``nedf3_processer/``, then cwd."""
    del config_path
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    normalized = _strip_leading_repo_dir_name(p, _REPO_ROOT)
    candidates = [
        _REPO_ROOT / normalized,
        _PACKAGE_ROOT / normalized,
        Path.cwd() / p,
        Path.cwd() / normalized,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return (_REPO_ROOT / normalized).resolve()


def _run_from_config(cfg: Dict[str, Any], config_path: Path) -> None:
    """Dispatch a NEDF3 -> LeRobot conversion from a parsed YAML config."""
    latency_cfg = _cfg_section(cfg, "latency")
    _cfg_require_keys(
        latency_cfg, ("wrist_cam_1", "wrist_cam_2", "global_cam"), "latency"
    )
    dataset_cfg = _cfg_section(cfg, "dataset")
    _cfg_require_keys(
        dataset_cfg,
        (
            "data_dir",
            "dataset_name",
            "output_dir",
            "task_desc",
            "fps",
            "ee_dim",
            "omit_ego_state_action",
        ),
        "dataset",
    )
    cameras_cfg = _cfg_section(cfg, "cameras")
    _cfg_require_keys(
        cameras_cfg,
        (
            "wrist_cam_1",
            "wrist_cam_2",
            "global_cam",
            "wrist_image_size",
            "global_image_size",
        ),
        "cameras",
    )
    selection_cfg = _cfg_section(cfg, "selection")
    _cfg_require_keys(selection_cfg, ("num_demos", "num_subsets", "seed"), "selection")
    gripper_cfg = _cfg_section(cfg, "gripper")
    _cfg_require_keys(gripper_cfg, ("left", "right", "module_root"), "gripper")
    for arm in ("left", "right"):
        if not isinstance(gripper_cfg[arm], dict):
            raise ValueError(
                f"Config gripper.{arm} must be a mapping (object), not a scalar."
            )

    if not str(dataset_cfg["data_dir"]).strip():
        raise ValueError("dataset.data_dir must be a non-empty path in the config YAML.")
    if not str(dataset_cfg["dataset_name"]).strip():
        raise ValueError(
            "dataset.dataset_name must be a non-empty string in the config YAML."
        )

    # ``latency.*`` values in YAML are milliseconds; convert to seconds for offsets.
    latency_wrist_cam_1 = float(latency_cfg["wrist_cam_1"]) / 1000.0
    latency_wrist_cam_2 = float(latency_cfg["wrist_cam_2"]) / 1000.0
    latency_global_cam = float(latency_cfg["global_cam"]) / 1000.0

    data_dir = _resolve_project_path(dataset_cfg["data_dir"], config_path)
    dataset_name = str(dataset_cfg["dataset_name"])
    output_dir = _resolve_project_path(dataset_cfg["output_dir"], config_path)
    task_desc = dataset_cfg["task_desc"]
    fps = int(dataset_cfg["fps"])
    ee_dim = int(dataset_cfg["ee_dim"])
    omit_ego_state_action = bool(dataset_cfg["omit_ego_state_action"])

    wrist_cam_1 = cameras_cfg["wrist_cam_1"]
    wrist_cam_2 = cameras_cfg["wrist_cam_2"]
    global_cam = cameras_cfg["global_cam"]
    wrist_image_size = _coerce_image_size(
        cameras_cfg["wrist_image_size"],
        "cameras.wrist_image_size",
    )
    global_image_size = _coerce_image_size(
        cameras_cfg["global_image_size"],
        "cameras.global_image_size",
    )

    num_demos = int(selection_cfg["num_demos"])
    num_subsets = int(selection_cfg["num_subsets"])
    subset_seed = int(selection_cfg["seed"])

    (
        gripper_cam_by_arm,
        gripper_masks_by_arm,
        gripper_cfg_by_arm,
        gripper_module_root,
        width_calibration_json,
        gripper_short_gap_interp_max_frames,
    ) = _resolve_gripper_config(gripper_cfg)

    gripper_masks_by_arm = {
        arm: str(_resolve_project_path(path_str, config_path))
        for arm, path_str in gripper_masks_by_arm.items()
    }
    gripper_module_root = _resolve_project_path(gripper_module_root, config_path)
    if not gripper_module_root.is_dir():
        raise FileNotFoundError(
            f"gripper.module_root is not a directory: {gripper_module_root}"
        )
    if not (gripper_module_root / "detector.py").is_file():
        raise FileNotFoundError(
            f"gripper.module_root must contain detector.py; missing at "
            f"{gripper_module_root / 'detector.py'}"
        )

    if width_calibration_json:
        gripper_width_calibration_path = _resolve_project_path(
            width_calibration_json, config_path
        )
    else:
        gripper_width_calibration_path = _default_width_calibration_json_path(
            config_path
        )
    if not gripper_width_calibration_path.is_file():
        raise FileNotFoundError(
            "gripper width calibration JSON not found: "
            f"{gripper_width_calibration_path}. Set gripper.width_calibration_json "
            "in the config YAML or place nmx_vla alongside the repo."
        )
    _load_visual_width_calibration(gripper_width_calibration_path)

    backup_umi_to_tcp_transform = _parse_umi_to_tcp_transform_from_config(
        cfg.get("umi_to_tcp_transform")
    )
    if backup_umi_to_tcp_transform:
        logger.info(
            "YAML backup umi_to_tcp_transform: %d float(s) (left/right 4×4 row-major)",
            len(backup_umi_to_tcp_transform),
        )
    else:
        logger.info(
            "No umi_to_tcp_transform in config; TCP remap only when metadata provides >=16 floats"
        )

    common_kwargs: Dict[str, Any] = {
        "latency_wrist_cam_1": latency_wrist_cam_1,
        "latency_wrist_cam_2": latency_wrist_cam_2,
        "latency_global_cam": latency_global_cam,
        "output_dir": output_dir,
        "wrist_cam_1": wrist_cam_1,
        "wrist_cam_2": wrist_cam_2,
        "global_cam": global_cam,
        "task_desc": task_desc,
        "fps": fps,
        "ee_dim": ee_dim,
        "omit_ego_state_action": omit_ego_state_action,
        "image_size": wrist_image_size,
        "global_image_size": global_image_size,
        "gripper_cam_by_arm": gripper_cam_by_arm,
        "gripper_masks_by_arm": gripper_masks_by_arm,
        "gripper_cfg_by_arm": gripper_cfg_by_arm,
        "gripper_module_root": gripper_module_root,
        "gripper_width_calibration_path": gripper_width_calibration_path,
        "gripper_short_gap_interp_max_frames": gripper_short_gap_interp_max_frames,
        "backup_umi_to_tcp_transform": backup_umi_to_tcp_transform,
    }

    if num_subsets > 1:
        if num_demos != 0:
            raise ValueError(
                f"When selection.num_subsets > 1, selection.num_demos must be 0 "
                f"to include all demos. Got num_demos={num_demos}."
            )

        all_demos = _list_demo_dirs(data_dir)
        subsets = _split_demos_to_subsets(all_demos, num_subsets, subset_seed)

        import multiprocessing as mp

        processes = []
        for i, subset_demos in enumerate(subsets):
            subset_name = f"{dataset_name}_{i:02d}"
            kwargs = {
                "data_dir": data_dir,
                **common_kwargs,
                "dataset_name": subset_name,
                "num_demos": 0,
                "pre_filtered_demos": subset_demos,
            }
            p = mp.Process(target=convert_nedf3_to_lerobot, kwargs=kwargs)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        failures: List[Tuple[str, Optional[int]]] = []
        for i, p in enumerate(processes):
            if p.exitcode != 0:
                failures.append((f"{dataset_name}_{i:02d}", p.exitcode))
        if failures:
            detail = ", ".join(
                f"{name} (exit={code})" if code is not None else f"{name} (exit unknown)"
                for name, code in failures
            )
            raise RuntimeError(
                f"{len(failures)} subset conversion process(es) failed: {detail}"
            )

    else:
        convert_nedf3_to_lerobot(
            data_dir=data_dir,
            dataset_name=dataset_name,
            num_demos=num_demos,
            **common_kwargs,
        )


def main() -> None:
    """CLI: load ``--config`` YAML and run conversion (see ``configs/nedf3_to_lerobot.yaml``)."""
    from argparse import ArgumentParser

    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = ArgumentParser(
        description="Convert NEDF3 (HAL) datasets to LeRobot v2.1."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the data-transfer YAML config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    logger.info("Loaded conversion config from %s", config_path)
    _run_from_config(cfg, config_path)


if __name__ == "__main__":
    main()
