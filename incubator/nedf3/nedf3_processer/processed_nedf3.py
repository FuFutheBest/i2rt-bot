"""
NEDF3 (HAL) offline processor: vision gripper width, latency shift, main-camera alignment.

Reads NEDF3 demos, applies UMI tracker→TCP remap, vision gripper width (detector +
``visual_angle_to_width.json``), per-camera latency shift, and writes aligned NEDF3
episodes. ``umi_to_tcp_transform`` is taken from episode metadata when present
(≥16 floats); otherwise from top-level ``umi_to_tcp_transform`` in
``configs/nedf3_processed.yaml``.

Usage (verified ``data-processing`` runtime profile)::

    python3 scripts/diagnostics/runtime_profiles.py run \\
        --role data-processing -- \\
        incubator/nedf3/nedf3_processer/processed_nedf3.py \\
        --config incubator/nedf3/nedf3_processer/configs/nedf3_processed.yaml

Author: Yichen Xie
"""

from __future__ import annotations

import io
import json
import logging
import math
import random
import shutil
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

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

NEDF3_POSE_TOPIC_BY_ARM: Dict[str, str] = {
    "left": "nmx/hal/umi/left/pose",
    "right": "nmx/hal/umi/right/pose",
    "ego": "nmx/hal/umi/ego/pose",
}

NEDF3_TCP_TOPIC_BY_ARM = NEDF3_POSE_TOPIC_BY_ARM

NEDF3_GRIPPER_TOPIC_BY_ARM: Dict[str, str] = {
    "left": "nmx/hal/umi/left/hand_state",
    "right": "nmx/hal/umi/right/hand_state",
}

NEDF3_CAMERA_TOPIC_FMT = "nmx/hal/camera/{camera_id}/rgbd"

OFFLINE_PROCESS_DEFAULT_CFG_JSON: str = """{
    "erode_px": 0,
    "min_contour_area": 1.5,
    "roi_padding": 40
}"""

_GRIPPER_DETECTOR_BY_ROOT: Dict[str, Any] = {}
_VISUAL_WIDTH_CALIBRATION: Any = None
_VISUAL_WIDTH_CALIBRATION_PATH: Optional[Path] = None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


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


def _strip_leading_repo_dir_name(relative: Path, repo_root: Path) -> Path:
    parts = relative.parts
    if parts and parts[0] == repo_root.name:
        return Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return relative


def _resolve_project_path(raw: Union[str, Path], config_path: Path) -> Path:
    """Resolve YAML paths: try SuperInference repo root, then ``nedf3_processer/``, then cwd."""
    del config_path  # anchor paths from package layout, not config file depth
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


def _resolve_gripper_config(
    gripper_section: Dict[str, Any],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Path, Optional[str]]:
    cam_by_arm: Dict[str, str] = {}
    masks_by_arm: Dict[str, str] = {}
    cfg_by_arm: Dict[str, str] = {}

    for arm in ("left", "right"):
        per_arm = gripper_section.get(arm)
        if not isinstance(per_arm, dict):
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
            "gripper.module_root must be set to a non-empty path (directory with detector.py)."
        )
    module_root = Path(str(raw_root).strip())

    raw_calibration = gripper_section.get("width_calibration_json")
    width_calibration_json = (
        str(raw_calibration).strip() if raw_calibration is not None else None
    )
    if width_calibration_json == "":
        width_calibration_json = None

    return cam_by_arm, masks_by_arm, cfg_by_arm, module_root, width_calibration_json


def _coerce_image_size(value: Any, field_name: str) -> Tuple[int, int]:
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
                    f"{field_name} height and width must be positive, got {(height, width)}."
                )
            return (height, width)
    raise ValueError(
        f"{field_name} expects an int, a 1-element list, or [height, width]; got {value!r}."
    )


# ---------------------------------------------------------------------------
# NEDF3 I/O helpers
# ---------------------------------------------------------------------------


def _extract_log_time_ns(msg: Any) -> int:
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
    return getattr(msg, "decoded_message", msg)


def _iter_topic_messages(reader: Any, topic: str) -> Iterator[Any]:
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
    for attr in (None, "rgb", "image"):
        container = decoded_msg if attr is None else getattr(decoded_msg, attr, None)
        if container is None:
            continue
        data = getattr(container, "data", None)
        if data:
            return np.frombuffer(data, dtype=np.uint8)
    return None


def _decode_rgb_buffer(rgb_buffer: np.ndarray) -> Optional[np.ndarray]:
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
    p = np.asarray(pose_xyzw, dtype=np.float64).reshape(7)
    return [
        float(p[0]),
        float(p[1]),
        float(p[2]),
        float(p[6]),
        float(p[3]),
        float(p[4]),
        float(p[5]),
    ]


def _row_major_4x4_to_mat(flat16: np.ndarray) -> np.ndarray:
    v = np.asarray(flat16, dtype=np.float64).reshape(16)
    return v.reshape(4, 4)


def _pose_vec_wxyz_to_T4(pose_wxyz: Union[List[float], np.ndarray]) -> np.ndarray:
    p = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    x, y, z, qw, qx, qy, qz = p
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = (x, y, z)
    return T


def _T4_to_pose_vec_wxyz(T: np.ndarray) -> List[float]:
    Rm = np.asarray(T, dtype=np.float64)[:3, :3]
    qx, qy, qz, qw = R.from_matrix(Rm).as_quat()
    x, y, z = T[:3, 3]
    return [float(x), float(y), float(z), float(qw), float(qx), float(qy), float(qz)]


def _read_umi_to_tcp_transform_flat(reader: Any, metadata_path: Path) -> List[float]:
    """Read ``umi_to_tcp_transform`` from NEDF3 reader metadata or ``metadata.json``."""
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
            arm_dict[ts] = _T4_to_pose_vec_wxyz(Tw_tr @ inv_m)


def _resolve_demo_root(demo_path: Path) -> Path:
    if (demo_path / "metadata.json").exists():
        return demo_path
    for candidate in sorted(
        [d for d in demo_path.glob("*") if d.is_dir() and not d.name.startswith(".")]
    ):
        if (candidate / "metadata.json").exists():
            return candidate
    raise FileNotFoundError(f"Could not find metadata.json in demo directory: {demo_path}")


def _list_demo_dirs(data_dir: Path) -> List[Path]:
    if (data_dir / "metadata.json").exists():
        return [data_dir]
    return sorted([d for d in data_dir.glob("*") if d.is_dir()], key=lambda x: x.name)


def _normalize_metadata_data_format(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if "." in s:
        return s.rsplit(".", 1)[-1]
    return s


def _load_nedf3_metadata(demo_path: Path) -> Tuple[Path, Dict[str, Any]]:
    demo_dir = _resolve_demo_root(demo_path)
    metadata_path = demo_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_dict = json.load(f)
    fmt = _normalize_metadata_data_format(metadata_dict.get("data_format"))
    if fmt != "NEDF3":
        raise ValueError(
            f"Expected NEDF3 format, got {metadata_dict.get('data_format')!r} "
            f"(normalized {fmt!r})"
        )
    return metadata_path, metadata_dict


def _infer_arm_names(topic_keys: List[str], include_ego: bool = False) -> List[str]:
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
    metadata_dict: Dict[str, Any],
    wrist_cam_1: Optional[str],
    wrist_cam_2: Optional[str],
    global_cam: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    camera_info = metadata_dict.get("camera_info") or {}
    normalized = {str(k): str(v) for k, v in camera_info.items()}
    left_wrist = wrist_cam_1 or _infer_camera_id(normalized, ["left", "wrist"])
    right_wrist = wrist_cam_2 or _infer_camera_id(normalized, ["right", "wrist"])
    global_view = (
        global_cam
        or _infer_camera_id(normalized, ["head"])
        or _infer_camera_id(normalized, ["global"])
    )
    return left_wrist, right_wrist, global_view


def _build_camera_keys(
    metadata_dict: Dict[str, Any],
    wrist_cam_1: Optional[str],
    wrist_cam_2: Optional[str],
    global_cam: Optional[str],
    wrist_image_size: Tuple[int, int],
    global_image_size: Tuple[int, int],
) -> Tuple[
    List[Tuple[str, str, Tuple[int, int]]],
    Optional[str],
    Optional[str],
    Optional[str],
]:
    wrist_cam_1, wrist_cam_2, global_cam = _resolve_camera_ids(
        metadata_dict, wrist_cam_1, wrist_cam_2, global_cam
    )
    camera_keys: List[Tuple[str, str, Tuple[int, int]]] = []
    if wrist_cam_1:
        camera_keys.append(
            ("observation.images.wrist_image_1", wrist_cam_1, wrist_image_size)
        )
    if global_cam:
        camera_keys.append(("observation.images.image", global_cam, global_image_size))
    if wrist_cam_2:
        camera_keys.append(
            ("observation.images.wrist_image_2", wrist_cam_2, wrist_image_size)
        )
    return camera_keys, wrist_cam_1, wrist_cam_2, global_cam


def _resize_image_to_target(img: np.ndarray, image_size: tuple) -> np.ndarray:
    target_h, target_w = image_size
    h, w = img.shape[:2]
    if target_h == target_w:
        if h > w:
            img = img[h // 2 - w // 2 : h // 2 + w // 2, :]
        else:
            img = img[:, w // 2 - h // 2 : w // 2 + h // 2]
        return cv2.resize(img, (target_w, target_h))
    if h * target_w == w * target_h:
        return cv2.resize(img, (target_w, target_h))
    raise ValueError(
        "Only support target image size with 1:1 ratio or the same aspect ratio as "
        f"the source image. Got target size {image_size} and source size {(h, w)}."
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
    camera_keys: List[Tuple[str, str, Tuple[int, int]]],
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, np.ndarray]]:
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
                "Camera topic '%s' missing in %s. Camera '%s' will be zeros.",
                topic,
                demo_dir,
                cam_id,
            )
        camera_data_by_key[cam_key] = images_by_ts
        camera_timestamps_by_key[cam_key] = (
            np.array(sorted(images_by_ts.keys())) if images_by_ts else np.array([])
        )

    return camera_data_by_key, camera_timestamps_by_key


def _shift_camera_streams_by_latency(
    camera_data_by_key: Dict[str, Dict[int, np.ndarray]],
    camera_keys: List[Tuple[str, str, Tuple[int, int]]],
    latency_wrist_cam_1_s: float,
    latency_wrist_cam_2_s: float,
    latency_global_cam_s: float,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, np.ndarray]]:
    latency_s_by_key = {
        "observation.images.wrist_image_1": latency_wrist_cam_1_s,
        "observation.images.wrist_image_2": latency_wrist_cam_2_s,
        "observation.images.image": latency_global_cam_s,
    }
    new_data: Dict[str, Dict[int, np.ndarray]] = {}
    new_ts: Dict[str, np.ndarray] = {}
    for cam_key, _, _ in camera_keys:
        offset_ns = int(round(float(latency_s_by_key.get(cam_key, 0.0)) * 1e9))
        old = camera_data_by_key.get(cam_key, {})
        if offset_ns == 0:
            shifted = dict(old)
        else:
            shifted = {int(ts) - offset_ns: img for ts, img in old.items()}
        new_data[cam_key] = shifted
        keys = shifted.keys()
        new_ts[cam_key] = (
            np.array(sorted(keys), dtype=np.int64) if keys else np.array([], dtype=np.int64)
        )
    return new_data, new_ts


def _observation_key_for_camera_id(
    camera_keys: List[Tuple[str, str, Tuple[int, int]]],
    camera_id: Optional[str],
) -> Optional[str]:
    if not camera_id or not str(camera_id).strip():
        return None
    cid = str(camera_id).strip()
    for cam_key, cam_dev_id, _ in camera_keys:
        if str(cam_dev_id) == cid:
            return cam_key
    return None


def _alignment_timestamps_ns(
    camera_timestamps_by_key: Dict[str, np.ndarray],
    camera_keys: List[Tuple[str, str, Tuple[int, int]]],
    metadata_dict: Dict[str, Any],
    demo_dir: Path,
) -> np.ndarray:
    main_raw = metadata_dict.get("main_camera")
    main_id = str(main_raw).strip() if main_raw is not None else ""
    if not main_id:
        raise ValueError(
            f"{demo_dir / 'metadata.json'}: required key 'main_camera' is missing or blank."
        )
    align_key = _observation_key_for_camera_id(camera_keys, main_id)
    if align_key is None:
        exported = [d for _, d, _ in camera_keys]
        raise ValueError(
            f"{demo_dir}: metadata main_camera={main_id!r} does not match exported ids {exported}."
        )
    main_ts = camera_timestamps_by_key.get(align_key)
    if main_ts is None or main_ts.size == 0:
        raise ValueError(
            f"{demo_dir}: main_camera={main_id!r} ({align_key!r}) has no decoded frames."
        )
    return main_ts


# ---------------------------------------------------------------------------
# Vision gripper width
# ---------------------------------------------------------------------------


def _import_gripper_detector(module_root: Path) -> Any:
    module_root = module_root.expanduser().resolve()
    cache_key = str(module_root)
    cached = _GRIPPER_DETECTOR_BY_ROOT.get(cache_key)
    if cached is not None:
        return cached

    det_path = module_root / "detector.py"
    if not det_path.is_file():
        raise FileNotFoundError(f"Gripper detector not found: expected {det_path}")

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
    if not cfg_value:
        return ""
    p = Path(cfg_value)
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return cfg_value


def _load_visual_width_calibration(calibration_path: Path) -> Any:
    global _VISUAL_WIDTH_CALIBRATION, _VISUAL_WIDTH_CALIBRATION_PATH
    resolved = calibration_path.expanduser().resolve()
    if (
        _VISUAL_WIDTH_CALIBRATION is not None
        and _VISUAL_WIDTH_CALIBRATION_PATH == resolved
    ):
        return _VISUAL_WIDTH_CALIBRATION

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
    if angle_deg is None or not math.isfinite(float(angle_deg)):
        return 0.0
    from visual_width_geometry import estimate_visual_width_from_angle_deg

    return float(
        estimate_visual_width_from_angle_deg(
            arm, float(angle_deg), calibration
        ).width_m
    )


def _detect_gripper_width_per_image(
    detector: Any,
    images_by_ts: Dict[int, np.ndarray],
    *,
    arm: str,
    calibration: Any,
) -> Dict[int, List[float]]:
    widths_by_ts: Dict[int, List[float]] = {}
    for ts in sorted(images_by_ts.keys()):
        bgr_img = cv2.cvtColor(images_by_ts[ts], cv2.COLOR_RGB2BGR)
        result = detector.process_for_export(bgr_img)
        width_m = _marker_angle_deg_to_width_m(
            arm, result.angle_deg, calibration=calibration
        )
        widths_by_ts[ts] = [width_m]
    return widths_by_ts


def _populate_gripper_from_vision(
    arm_to_cam_key: Dict[str, str],
    masks_dir_by_arm: Dict[str, str],
    cfg_json_by_arm: Dict[str, str],
    camera_data_by_key: Dict[str, Dict[int, np.ndarray]],
    state_data_by_arm: Dict[str, Dict[int, List[float]]],
    state_timestamps_by_arm: Dict[str, np.ndarray],
    gripper_module_root: Path,
    width_calibration_path: Path,
    demo_dir: Path,
) -> None:
    """Fill gripper dicts with physical width in meters from vision."""
    if not arm_to_cam_key:
        return

    GripperDetector = _import_gripper_detector(gripper_module_root)
    calibration = _load_visual_width_calibration(width_calibration_path)

    for arm, cam_key in arm_to_cam_key.items():
        masks_dir = masks_dir_by_arm.get(arm)
        if not masks_dir:
            continue
        if cam_key not in camera_data_by_key:
            logger.warning(
                "Vision gripper for arm '%s': camera key '%s' not loaded in %s.",
                arm,
                cam_key,
                demo_dir,
            )
            continue
        images_by_ts = camera_data_by_key[cam_key]
        if not images_by_ts:
            logger.warning(
                "Vision gripper for arm '%s': no images on %s in %s.",
                arm,
                cam_key,
                demo_dir,
            )
            continue

        cfg_json = _resolve_cfg_json(cfg_json_by_arm.get(arm))
        if not cfg_json:
            cfg_json = OFFLINE_PROCESS_DEFAULT_CFG_JSON
        detector = GripperDetector(masks_dir, cfg_json)
        detector.reset()
        widths = _detect_gripper_width_per_image(
            detector,
            images_by_ts,
            arm=arm,
            calibration=calibration,
        )
        state_data_by_arm[arm] = widths
        state_timestamps_by_arm[arm] = np.array(sorted(widths.keys()))
        valid = [row[0] for row in widths.values() if row]
        if valid:
            logger.info(
                "Vision gripper for arm '%s': %d width samples from %s in %s "
                "(m range %.4f .. %.4f)",
                arm,
                len(widths),
                cam_key,
                demo_dir,
                min(valid),
                max(valid),
            )
        else:
            logger.info(
                "Vision gripper for arm '%s': %d width samples from %s in %s",
                arm,
                len(widths),
                cam_key,
                demo_dir,
            )


# ---------------------------------------------------------------------------
# TCP interpolation and per-frame alignment
# ---------------------------------------------------------------------------


def _find_nearest_timestamp(
    target_ts: Union[int, float], available_timestamps: np.ndarray
) -> Union[int, float]:
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
    return si.interp1d(
        t, x, axis=0, bounds_error=False, fill_value=(x[0], x[-1])
    )


class TcpPoseInterpolator:
    """Interpolate 7D TCP ``[x,y,z,qw,qx,qy,qz]`` in log time (ns)."""

    def __init__(self, t_ns: np.ndarray, poses_wxyz: np.ndarray) -> None:
        self._t = np.asarray(t_ns, dtype=np.float64).reshape(-1)
        self._poses = np.asarray(poses_wxyz, dtype=np.float64)
        if self._t.size != self._poses.shape[0]:
            raise ValueError(
                f"TcpPoseInterpolator: len(t)={self._t.size} != rows={self._poses.shape[0]}"
            )
        if self._t.size == 0:
            self._t0 = 0.0
            self._mode = "empty"
            return
        self._t0 = float(self._t[0])
        self._t = self._t - self._t0
        if self._t.size == 1:
            self._mode = "constant"
            self._const = self._poses[0].copy()
            return
        pos = self._poses[:, :3]
        qw, qx, qy, qz = (
            self._poses[:, 3],
            self._poses[:, 4],
            self._poses[:, 5],
            self._poses[:, 6],
        )
        rot = R.from_quat(np.column_stack([qx, qy, qz, qw]))
        self._pos_i = _interp1d_axis0(self._t, pos)
        self._rot_s = Slerp(self._t, rot)
        self._mode = "interp"

    def sample(self, t_query_ns: Union[int, float, np.number]) -> List[float]:
        if self._mode == "empty":
            return [0.0] * 7
        if self._mode == "constant":
            return [float(x) for x in self._const]
        tq = float(t_query_ns) - self._t0
        tq = min(max(tq, float(self._t[0])), float(self._t[-1]))
        pos = self._pos_i(np.array([tq], dtype=np.float64))[0]
        qx, qy, qz, qw = self._rot_s(np.array([tq], dtype=np.float64)).as_quat()[0]
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
        if not rows:
            continue
        out[arm_name] = TcpPoseInterpolator(
            np.asarray(times, dtype=np.float64), np.stack(rows, axis=0)
        )
    return out


def _align_camera_image(
    ts: Union[int, float],
    cam_key: str,
    target_size: Tuple[int, int],
    camera_data_by_key: Dict[str, Dict[int, np.ndarray]],
    camera_timestamps_by_key: Dict[str, np.ndarray],
) -> np.ndarray:
    camera_timestamps = camera_timestamps_by_key[cam_key]
    if camera_timestamps.size > 0:
        nearest_cam_ts = _find_nearest_timestamp(ts, camera_timestamps)
        img = camera_data_by_key[cam_key].get(int(nearest_cam_ts))
        if img is not None:
            return _resize_image_to_target(img, target_size)
    return np.zeros((*target_size, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# NEDF3 writer
# ---------------------------------------------------------------------------


def _pose_wxyz_to_hal_xyzw(pose_wxyz: Union[List[float], np.ndarray]) -> List[float]:
    p = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    return [
        float(p[0]),
        float(p[1]),
        float(p[2]),
        float(p[4]),
        float(p[5]),
        float(p[6]),
        float(p[3]),
    ]


def _write_lowdim(
    writer: Any, topic: str, values: List[float], timestamp_ns: int
) -> None:
    from nmx_msg.Lowdim_pb2 import LowdimData

    msg = LowdimData()
    msg.data.extend([float(v) for v in values])
    writer.write_message(topic, msg, timestamp_ns, timestamp_ns)


def _write_rgbd_jpeg(
    writer: Any,
    camera_id: str,
    rgb_img: np.ndarray,
    timestamp_ns: int,
) -> None:
    from nmx_msg.Image_pb2 import Image, ImageDataType

    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    ok, jpg_bytes = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return
    topic = NEDF3_CAMERA_TOPIC_FMT.format(camera_id=camera_id)
    image_msg = Image()
    image_msg.data = jpg_bytes.tobytes()
    image_msg.type = ImageDataType.ENCODED
    image_msg.encoded_format = "jpeg"
    image_msg.rows = int(bgr.shape[0])
    image_msg.cols = int(bgr.shape[1])
    image_msg.channels = int(bgr.shape[2]) if bgr.ndim == 3 else 1
    writer.write_message(topic, image_msg, timestamp_ns, timestamp_ns)


def _gripper_value_at_ts(
    arm: str,
    ts: int,
    state_data_by_arm: Dict[str, Dict[int, List[float]]],
    state_timestamps_by_arm: Dict[str, np.ndarray],
) -> float:
    state_timestamps = state_timestamps_by_arm.get(arm, np.array([]))
    if state_timestamps.size == 0:
        return 0.0
    nearest_ts = _find_nearest_timestamp(ts, state_timestamps)
    gripper = state_data_by_arm.get(arm, {}).get(int(nearest_ts), [0.0])
    return float(gripper[0]) if gripper else 0.0


def _write_aligned_episode_to_nedf3(
    writer: Any,
    timestamps: np.ndarray,
    camera_keys: List[Tuple[str, str, Tuple[int, int]]],
    pose_interp_by_arm: Dict[str, TcpPoseInterpolator],
    state_data_by_arm: Dict[str, Dict[int, List[float]]],
    state_timestamps_by_arm: Dict[str, np.ndarray],
    camera_data_by_key: Dict[str, Dict[int, np.ndarray]],
    camera_timestamps_by_key: Dict[str, np.ndarray],
    tcp_arms: List[str],
    gripper_arms: List[str],
) -> int:
    count = 0
    for ts in timestamps:
        ts_int = int(ts)
        for arm in tcp_arms:
            topic = NEDF3_TCP_TOPIC_BY_ARM.get(arm)
            interp = pose_interp_by_arm.get(arm)
            if not topic or interp is None:
                continue
            _write_lowdim(
                writer,
                topic,
                _pose_wxyz_to_hal_xyzw(interp.sample(ts_int)),
                ts_int,
            )

        for arm in gripper_arms:
            topic = NEDF3_GRIPPER_TOPIC_BY_ARM.get(arm)
            if not topic:
                continue
            _write_lowdim(
                writer,
                topic,
                [_gripper_value_at_ts(arm, ts_int, state_data_by_arm, state_timestamps_by_arm)],
                ts_int,
            )

        for cam_key, cam_id, target_size in camera_keys:
            img = _align_camera_image(
                ts_int,
                cam_key,
                target_size,
                camera_data_by_key,
                camera_timestamps_by_key,
            )
            if img is None or img.size == 0:
                continue
            _write_rgbd_jpeg(writer, cam_id, img, ts_int)
        count += 1
    return count


def _infer_processing_arms(
    topic_names: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Return ``(tcp_arms, gripper_arms, state_arm_names)`` from HAL topics.

    Gripper width is vision-only on left/right; ego has TCP only (no gripper).
    """
    all_arms = _infer_arm_names(topic_names, include_ego=True)
    tcp_arms = [a for a in ("left", "right", "ego") if a in all_arms]
    gripper_arms = [a for a in ("left", "right") if a in all_arms]
    return tcp_arms, gripper_arms, list(tcp_arms)


def _split_demos_to_subsets(
    demos: List[Path], num_subsets: int, seed: int
) -> List[List[Path]]:
    random.seed(seed)
    shuffled = demos.copy()
    random.shuffle(shuffled)
    return [shuffled[i::num_subsets] for i in range(num_subsets)]


# ---------------------------------------------------------------------------
# Pipeline entry points
# ---------------------------------------------------------------------------


def process_demo_to_nedf3(
    demo_dir: Path,
    output_root: Path,
    camera_keys: List[Tuple[str, str, Tuple[int, int]]],
    gripper_cam_by_arm: Dict[str, str],
    gripper_masks_by_arm: Dict[str, str],
    gripper_cfg_by_arm: Dict[str, str],
    gripper_module_root: Path,
    gripper_width_calibration_path: Path,
    backup_umi_to_tcp_transform: List[float],
    latency_wrist_cam_1_s: float,
    latency_wrist_cam_2_s: float,
    latency_global_cam_s: float,
) -> int:
    from nmx_nedf_api import NEDFFactory, NEDFReaderConfig, NEDFType, NEDFTopicsConfig, NEDFWriterConfig

    demo_dir = _resolve_demo_root(demo_dir)
    src_metadata_path = demo_dir / "metadata.json"

    out_demo_dir = output_root / demo_dir.name
    if out_demo_dir.exists():
        shutil.rmtree(out_demo_dir)
    out_demo_dir.mkdir(parents=True, exist_ok=True)
    out_metadata_path = out_demo_dir / "metadata.json"
    shutil.copy2(src_metadata_path, out_metadata_path)

    src_reader = NEDFFactory.get_reader(
        NEDFReaderConfig(metadata_file_path=str(src_metadata_path))
    )
    topic_names = list(src_reader.get_raw_topics_timestamps().keys())

    tcp_arms, gripper_arms, state_arm_names = _infer_processing_arms(topic_names)
    if not tcp_arms:
        raise ValueError(f"No TCP arms found in {demo_dir}")

    (
        pose_data_by_arm,
        pose_timestamps_by_arm,
        state_data_by_arm,
        state_timestamps_by_arm,
    ) = _load_arm_streams(
        reader=src_reader,
        topic_keys=topic_names,
        demo_dir=demo_dir,
        state_arm_names=state_arm_names,
        action_arm_names=list(state_arm_names),
    )

    umi_flat, umi_source = _resolve_umi_to_tcp_transform_flat(
        src_reader,
        src_metadata_path,
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
        reader=src_reader,
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

    _populate_gripper_from_vision(
        arm_to_cam_key=gripper_cam_by_arm,
        masks_dir_by_arm=gripper_masks_by_arm,
        cfg_json_by_arm=gripper_cfg_by_arm,
        camera_data_by_key=camera_data_by_key,
        state_data_by_arm=state_data_by_arm,
        state_timestamps_by_arm=state_timestamps_by_arm,
        gripper_module_root=gripper_module_root,
        width_calibration_path=gripper_width_calibration_path,
        demo_dir=demo_dir,
    )

    with open(src_metadata_path, "r", encoding="utf-8") as mf:
        metadata_dict_demo: Dict[str, Any] = json.load(mf)

    timestamps = _alignment_timestamps_ns(
        camera_timestamps_by_key,
        camera_keys,
        metadata_dict_demo,
        demo_dir,
    )
    pose_interp_by_arm = _build_tcp_pose_interpolators_by_arm(
        pose_data_by_arm, pose_timestamps_by_arm
    )

    writer_config = NEDFWriterConfig(
        metadata_file_path=out_metadata_path,
        metadata=src_reader.metadata,
        topics_config=NEDFTopicsConfig(),
        max_file_size_mb=100,
        mcap_chunk_size_kb=256,
    )

    with NEDFFactory.get_writer(NEDFType.NEDF3, writer_config) as writer:
        frame_count = _write_aligned_episode_to_nedf3(
            writer=writer,
            timestamps=timestamps,
            camera_keys=camera_keys,
            pose_interp_by_arm=pose_interp_by_arm,
            state_data_by_arm=state_data_by_arm,
            state_timestamps_by_arm=state_timestamps_by_arm,
            camera_data_by_key=camera_data_by_key,
            camera_timestamps_by_key=camera_timestamps_by_key,
            tcp_arms=tcp_arms,
            gripper_arms=gripper_arms,
        )

    logger.info(
        "Wrote processed NEDF3 %s -> %s (%d frames)",
        demo_dir.name,
        out_demo_dir,
        frame_count,
    )
    return frame_count


def convert_nedf3_processed(
    data_dir: Path,
    output_dir: Path,
    *,
    wrist_cam_1: Optional[str],
    wrist_cam_2: Optional[str],
    global_cam: Optional[str],
    wrist_image_size: Tuple[int, int],
    global_image_size: Tuple[int, int],
    gripper_cam_by_arm: Dict[str, str],
    gripper_masks_by_arm: Dict[str, str],
    gripper_cfg_by_arm: Dict[str, str],
    gripper_module_root: Path,
    gripper_width_calibration_path: Path,
    backup_umi_to_tcp_transform: List[float],
    latency_wrist_cam_1_s: float,
    latency_wrist_cam_2_s: float,
    latency_global_cam_s: float,
    num_demos: int = 0,
    pre_filtered_demos: Optional[List[Path]] = None,
) -> None:
    import av

    av.logging.set_level(av.logging.ERROR)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    demos = pre_filtered_demos if pre_filtered_demos is not None else _list_demo_dirs(data_dir)
    if num_demos > 0:
        demos = demos[:num_demos]
    if not demos:
        raise ValueError(f"No demo directories found in {data_dir}")

    _, metadata_dict = _load_nedf3_metadata(demos[0])
    camera_keys, wrist_cam_1, wrist_cam_2, global_cam = _build_camera_keys(
        metadata_dict=metadata_dict,
        wrist_cam_1=wrist_cam_1,
        wrist_cam_2=wrist_cam_2,
        global_cam=global_cam,
        wrist_image_size=wrist_image_size,
        global_image_size=global_image_size,
    )

    cam_key_set = {ck for ck, _, _ in camera_keys}
    missing_masks = [
        arm
        for arm in ("left", "right")
        if not gripper_masks_by_arm.get(arm)
    ]
    if missing_masks:
        raise ValueError(
            f"Vision gripper masks required for arm(s): {missing_masks}. "
            "Set gripper.<arm>.masks_dir in YAML."
        )
    for arm, cam_key in gripper_cam_by_arm.items():
        if cam_key not in cam_key_set:
            raise ValueError(
                f"gripper.{arm}.camera={cam_key!r} not in exported cameras {sorted(cam_key_set)}."
            )

    logger.info(
        "Cameras: wrist_1=%s global=%s wrist_2=%s sizes wrist=%s global=%s",
        wrist_cam_1,
        global_cam,
        wrist_cam_2,
        wrist_image_size,
        global_image_size,
    )
    logger.info("Output NEDF3 root: %s", output_dir)

    for demo_dir in tqdm(demos, desc="Processing demos"):
        process_demo_to_nedf3(
            demo_dir=demo_dir,
            output_root=output_dir,
            camera_keys=camera_keys,
            gripper_cam_by_arm=gripper_cam_by_arm,
            gripper_masks_by_arm=gripper_masks_by_arm,
            gripper_cfg_by_arm=gripper_cfg_by_arm,
            gripper_module_root=gripper_module_root,
            gripper_width_calibration_path=gripper_width_calibration_path,
            backup_umi_to_tcp_transform=backup_umi_to_tcp_transform,
            latency_wrist_cam_1_s=latency_wrist_cam_1_s,
            latency_wrist_cam_2_s=latency_wrist_cam_2_s,
            latency_global_cam_s=latency_global_cam_s,
        )

    logger.info("NEDF3 processing complete: %s", output_dir)


def _run_from_config(cfg: Dict[str, Any], config_path: Path) -> None:
    latency_cfg = _cfg_section(cfg, "latency")
    _cfg_require_keys(
        latency_cfg, ("wrist_cam_1", "wrist_cam_2", "global_cam"), "latency"
    )
    dataset_cfg = _cfg_section(cfg, "dataset")
    _cfg_require_keys(dataset_cfg, ("data_dir", "output_dir"), "dataset")
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

    latency_wrist_cam_1_s = float(latency_cfg["wrist_cam_1"]) / 1000.0
    latency_wrist_cam_2_s = float(latency_cfg["wrist_cam_2"]) / 1000.0
    latency_global_cam_s = float(latency_cfg["global_cam"]) / 1000.0

    data_dir = _resolve_project_path(dataset_cfg["data_dir"], config_path)
    output_dir = _resolve_project_path(dataset_cfg["output_dir"], config_path)

    num_demos = int(dataset_cfg.get("num_demos", selection_cfg["num_demos"]))
    num_subsets = int(selection_cfg["num_subsets"])
    subset_seed = int(selection_cfg["seed"])

    wrist_image_size = _coerce_image_size(
        cameras_cfg["wrist_image_size"], "cameras.wrist_image_size"
    )
    global_image_size = _coerce_image_size(
        cameras_cfg["global_image_size"], "cameras.global_image_size"
    )

    (
        gripper_cam_by_arm,
        gripper_masks_by_arm,
        gripper_cfg_by_arm,
        gripper_module_root,
        width_calibration_json,
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
            f"gripper.module_root must contain detector.py: {gripper_module_root / 'detector.py'}"
        )

    if width_calibration_json:
        gripper_width_calibration_path = _resolve_project_path(
            width_calibration_json, config_path
        )
    else:
        gripper_width_calibration_path = (
            _PACKAGE_ROOT / "configs" / "visual_angle_to_width.json"
        ).resolve()
    if not gripper_width_calibration_path.is_file():
        raise FileNotFoundError(
            "gripper width calibration JSON not found: "
            f"{gripper_width_calibration_path}. Set gripper.width_calibration_json."
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
        logger.warning(
            "No umi_to_tcp_transform in config; TCP remap only when metadata provides >=16 floats"
        )

    common_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "wrist_cam_1": cameras_cfg["wrist_cam_1"],
        "wrist_cam_2": cameras_cfg["wrist_cam_2"],
        "global_cam": cameras_cfg["global_cam"],
        "wrist_image_size": wrist_image_size,
        "global_image_size": global_image_size,
        "gripper_cam_by_arm": gripper_cam_by_arm,
        "gripper_masks_by_arm": gripper_masks_by_arm,
        "gripper_cfg_by_arm": gripper_cfg_by_arm,
        "gripper_module_root": gripper_module_root,
        "gripper_width_calibration_path": gripper_width_calibration_path,
        "backup_umi_to_tcp_transform": backup_umi_to_tcp_transform,
        "latency_wrist_cam_1_s": latency_wrist_cam_1_s,
        "latency_wrist_cam_2_s": latency_wrist_cam_2_s,
        "latency_global_cam_s": latency_global_cam_s,
        "num_demos": num_demos,
    }

    if num_subsets > 1:
        if num_demos != 0:
            raise ValueError(
                "When selection.num_subsets > 1, num_demos must be 0 (process all demos)."
            )
        subsets = _split_demos_to_subsets(
            _list_demo_dirs(data_dir), num_subsets, subset_seed
        )
        import multiprocessing as mp

        processes = []
        for i, subset_demos in enumerate(subsets):
            kwargs = {
                "data_dir": data_dir,
                **common_kwargs,
                "output_dir": output_dir / f"subset_{i:02d}",
                "pre_filtered_demos": subset_demos,
                "num_demos": 0,
            }
            p = mp.Process(target=convert_nedf3_processed, kwargs=kwargs)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        failures = [(i, p.exitcode) for i, p in enumerate(processes) if p.exitcode != 0]
        if failures:
            raise RuntimeError(f"Subset worker(s) failed: {failures}")
    else:
        convert_nedf3_processed(data_dir=data_dir, **common_kwargs)


def main() -> None:
    from argparse import ArgumentParser

    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = ArgumentParser(
        description=(
            "Process NEDF3 demos (vision gripper width, latency, align) -> NEDF3 output."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config (see configs/nedf3_processed.yaml).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    logger.info("Loaded config from %s", config_path)
    _run_from_config(cfg, config_path)


if __name__ == "__main__":
    main()
