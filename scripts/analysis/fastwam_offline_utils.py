"""Minimal utilities shared by read-only FastWAM offline evaluation tools."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from utils.calibration_utils import load_tcp_umi_transforms
from utils.superoffline_observation import array_keys_from_index_row


DEFAULT_GRIPPER_MAX_WIDTH = 0.0925319933860716


def parse_inference_indices(value: str) -> list[int]:
    """Parse the executor's explicit, ordered batch selection."""
    indices: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        index = int(item)
        if index < 0:
            raise ValueError("inference-indices must contain non-negative integers")
        if index in indices:
            raise ValueError("inference-indices must not contain duplicates")
        indices.append(index)
    if not indices:
        raise ValueError("inference-indices must contain at least one index")
    return indices


def batch_child_argv(
    argv: Sequence[str],
    *,
    inference_index: int,
    output_dir: Path,
) -> list[str]:
    """Replace batch/output flags for one in-process evaluator iteration."""
    result: list[str] = []
    skip_value = False
    for argument in argv:
        if skip_value:
            skip_value = False
            continue
        if argument in {"--inference-indices", "--output-dir", "--inference-index"}:
            skip_value = True
            continue
        if any(
            argument.startswith(prefix)
            for prefix in (
                "--inference-indices=",
                "--output-dir=",
                "--inference-index=",
            )
        ):
            continue
        result.append(argument)
    result.extend(
        [
            "--inference-index",
            str(inference_index),
            "--output-dir",
            str(output_dir),
        ]
    )
    return result


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass
class RecordedFrame:
    timestamp_ns: int
    file: Path
    array_keys: dict[str, str]
    device_timestamps_ns: dict[str, int]


@dataclass
class RecordedInference:
    index: int
    timestamp_ns: int
    file: Path
    array_keys: dict[str, str]
    device_timestamps_ns: dict[str, int]
    episode_start_timestamp_ns: int | None


class ObservationDataset:
    """Read the subset of TimestampedObservationRecorder data used offline."""

    def __init__(
        self,
        path: str | Path,
        *,
        device_aliases: Mapping[str, str] | None = None,
    ) -> None:
        requested = Path(path).expanduser().resolve()
        if requested.is_file() and requested.name == "manifest.json":
            requested = requested.parent
        if not requested.is_dir():
            raise FileNotFoundError(f"Observation directory does not exist: {requested}")
        self.root = requested
        self._device_aliases = {
            str(target).strip(): str(source).strip()
            for target, source in dict(device_aliases or {}).items()
            if str(target).strip() and str(source).strip()
        }
        manifest_path = self.root / "manifest.json"
        self.manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        self.frames = [
            RecordedFrame(
                timestamp_ns=int(
                    row.get("timestamp_ns", row.get("summary_timestamp_ns", 0))
                ),
                file=(self.root / str(row["file"])).resolve(),
                array_keys=array_keys_from_index_row(row, kind="frame"),
                device_timestamps_ns={
                    str(key): int(value)
                    for key, value in dict(
                        row.get("device_timestamps_ns", {})
                    ).items()
                },
            )
            for row in _read_jsonl(self.root / "frames.jsonl")
            if int(row.get("timestamp_ns", row.get("summary_timestamp_ns", 0)))
            > 0
            and (self.root / str(row.get("file", ""))).is_file()
        ]
        self.frames.sort(key=lambda item: item.timestamp_ns)
        if not self.frames:
            raise ValueError(f"No raw frames found in {self.root}")

        self.inferences = []
        for row in _read_jsonl(self.root / "inference.jsonl"):
            timestamp_ns = int(
                row.get("timestamp_ns", row.get("observation_timestamp_ns", 0))
            )
            file_path = (self.root / str(row.get("file", ""))).resolve()
            if timestamp_ns <= 0 or not file_path.is_file():
                continue
            array_keys = array_keys_from_index_row(row, kind="inference")
            self.inferences.append(
                RecordedInference(
                    index=int(row.get("index", len(self.inferences))),
                    timestamp_ns=timestamp_ns,
                    file=file_path,
                    array_keys=array_keys,
                    device_timestamps_ns={
                        str(key): int(value)
                        for key, value in dict(
                            row.get("device_timestamps_ns", {})
                        ).items()
                    },
                    episode_start_timestamp_ns=(
                        None
                        if row.get("episode_start_timestamp_ns") is None
                        else int(row["episode_start_timestamp_ns"])
                    ),
                )
            )
        self.inferences.sort(key=lambda item: (item.timestamp_ns, item.index))

    @staticmethod
    def _load_npz(path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as data:
            return {str(key): np.array(data[key], copy=True) for key in data.files}

    @staticmethod
    def _find_array(
        data: Mapping[str, np.ndarray], device_name: str
    ) -> np.ndarray:
        exact = data.get(device_name)
        if exact is not None:
            return exact
        raise KeyError(f"Recorded input is missing {device_name}; keys={list(data)}")

    def _source_device_name(self, target_device_name: str) -> str:
        return self._device_aliases.get(target_device_name, target_device_name)

    def _find_device_array(
        self,
        data: Mapping[str, np.ndarray],
        target_device_name: str,
    ) -> np.ndarray:
        return self._find_array(
            data,
            self._source_device_name(target_device_name),
        )

    def _device_timestamp_ns(
        self,
        timestamps: Mapping[str, int],
        target_device_name: str,
        fallback_ns: int,
    ) -> int:
        return int(
            timestamps.get(self._source_device_name(target_device_name), fallback_ns)
        )

    @staticmethod
    def _latest_camera(array: np.ndarray) -> np.ndarray:
        value = np.asarray(array)
        if value.ndim >= 4:
            value = value[-1]
        if value.ndim != 3:
            raise ValueError(f"Recorded camera input must be HWC, got {value.shape}")
        return np.ascontiguousarray(value.astype(np.uint8, copy=False))

    @staticmethod
    def _latest_robot_state(array: np.ndarray) -> np.ndarray:
        value = np.asarray(array)
        if value.ndim >= 2:
            value = value[-1]
        value = value.reshape(-1)
        if value.size < 44:
            raise ValueError(f"Robot input must have at least 44 values, got {value.shape}")
        return value

    def _load_inference(self, inference: RecordedInference) -> dict[str, np.ndarray]:
        arrays = self._load_npz(inference.file)
        if not inference.array_keys:
            return arrays
        return {
            name: arrays[array_key]
            for name, array_key in inference.array_keys.items()
            if array_key in arrays
        }

    def _load_frame(self, frame: RecordedFrame) -> dict[str, np.ndarray]:
        arrays = self._load_npz(frame.file)
        if not frame.array_keys:
            return arrays
        return {
            name: arrays[array_key]
            for name, array_key in frame.array_keys.items()
            if array_key in arrays
        }

    def _policy_observation_from_values(
        self,
        values: Mapping[str, np.ndarray],
        *,
        camera_devices: Sequence[str],
        robot_device: str,
    ) -> dict[str, np.ndarray]:
        """Normalize one raw recorder sample to a live policy observation."""
        observation = {
            name: np.array(
                self._latest_camera(self._find_device_array(values, name)), copy=True
            )
            for name in camera_devices
        }
        observation[robot_device] = np.array(
            self._latest_robot_state(self._find_device_array(values, robot_device)),
            copy=True,
        )
        return observation

    def _nearest_master_frame(
        self,
        target_ns: int,
        *,
        robot_device: str,
        anchor_ns: int,
        minimum_ns: int | None,
    ) -> RecordedFrame | None:
        timeline = [
            (
                self._device_timestamp_ns(
                    frame.device_timestamps_ns,
                    robot_device,
                    frame.timestamp_ns,
                ),
                frame,
            )
            for frame in self.frames
            if self._device_timestamp_ns(
                frame.device_timestamps_ns,
                robot_device,
                frame.timestamp_ns,
            )
            <= anchor_ns
            and (
                minimum_ns is None
                or self._device_timestamp_ns(
                    frame.device_timestamps_ns,
                    robot_device,
                    frame.timestamp_ns,
                )
                >= minimum_ns
            )
        ]
        if not timeline:
            return None
        return min(timeline, key=lambda item: abs(item[0] - target_ns))[1]

    def inference_observation_at(
        self,
        inference_index: int,
        *,
        history_frames: int,
        history_stride: int,
        camera_devices: Sequence[str],
        robot_device: str,
    ) -> tuple[list[list[np.ndarray]], np.ndarray, list[int], int]:
        if inference_index < 0 or inference_index >= len(self.inferences):
            raise IndexError(f"inference_index={inference_index} is out of range")
        inference = self.inferences[inference_index]
        anchor_master_ns = self._device_timestamp_ns(
            inference.device_timestamps_ns,
            robot_device,
            inference.timestamp_ns,
        )
        exact_values = self._load_inference(inference)
        latest_images = [
            self._latest_camera(self._find_device_array(exact_values, name))
            for name in camera_devices
        ]
        robot = self._latest_robot_state(
            self._find_device_array(exact_values, robot_device)
        )
        state = np.concatenate([robot[14:22], robot[36:44]]).astype(np.float64)

        histories: list[list[np.ndarray]] = []
        timestamps: list[int] = []
        period_ns = int(round(1e9 / 30.0))
        for offset in range(history_frames - 1, 0, -1):
            target_ns = anchor_master_ns - offset * history_stride * period_ns
            frame = self._nearest_master_frame(
                target_ns,
                robot_device=robot_device,
                anchor_ns=anchor_master_ns,
                minimum_ns=inference.episode_start_timestamp_ns,
            )
            if frame is None:
                histories.append([image.copy() for image in latest_images])
                timestamps.append(inference.timestamp_ns)
                continue
            values = self._load_frame(frame)
            histories.append(
                [
                    self._latest_camera(self._find_device_array(values, name))
                    for name in camera_devices
                ]
            )
            timestamps.append(frame.timestamp_ns)
        histories.append(latest_images)
        timestamps.append(inference.timestamp_ns)
        return histories, state, timestamps, inference.timestamp_ns

    def exact_policy_observation_at(
        self,
        inference_index: int,
        *,
        camera_devices: Sequence[str],
        robot_device: str,
    ) -> tuple[dict[str, np.ndarray], int]:
        """Return one recorder input in the shape consumed by a live policy.

        Offline evaluators must read the exact recorded input rather than attach
        to summary shared memory.  The camera and robot names are supplied by
        the selected native config so this method does not guess hardware ids.
        """
        if inference_index < 0 or inference_index >= len(self.inferences):
            raise IndexError(f"inference_index={inference_index} is out of range")
        inference = self.inferences[inference_index]
        values = self._load_inference(inference)
        return (
            self._policy_observation_from_values(
                values,
                camera_devices=camera_devices,
                robot_device=robot_device,
            ),
            inference.timestamp_ns,
        )

    def policy_observation_history_at(
        self,
        inference_index: int,
        *,
        history_frames: int,
        history_stride: int,
        observation_interval_s: float,
        camera_devices: Sequence[str],
        robot_device: str,
    ) -> tuple[dict[str, list[np.ndarray]], list[int], int]:
        """Rebuild a target policy's chronological observation history.

        The recorded inference supplies the exact latest values. Earlier values
        are selected from raw synchronized recorder frames using the same
        horizon, stride, and cadence owned by the target policy. This prevents a
        comparison from inheriting the source policy's history shape.
        """
        if isinstance(history_frames, bool) or int(history_frames) < 1:
            raise ValueError("history_frames must be a positive integer")
        if isinstance(history_stride, bool) or int(history_stride) < 1:
            raise ValueError("history_stride must be a positive integer")
        if not np.isfinite(observation_interval_s) or observation_interval_s <= 0:
            raise ValueError("observation_interval_s must be positive and finite")
        if inference_index < 0 or inference_index >= len(self.inferences):
            raise IndexError(f"inference_index={inference_index} is out of range")

        history_frames = int(history_frames)
        history_stride = int(history_stride)
        interval_ns = int(round(float(observation_interval_s) * 1e9))
        if interval_ns < 1:
            raise ValueError("observation_interval_s is too small to represent")

        inference = self.inferences[inference_index]
        latest, observation_timestamp_ns = self.exact_policy_observation_at(
            inference_index,
            camera_devices=camera_devices,
            robot_device=robot_device,
        )
        anchor_master_ns = self._device_timestamp_ns(
            inference.device_timestamps_ns,
            robot_device,
            observation_timestamp_ns,
        )
        samples = [latest]
        timestamps = [observation_timestamp_ns]

        # Match PolicyConnector's 50 ms historical-frame tolerance. When an
        # older frame is unavailable, it pads from the closest newer sample.
        for offset in range(1, history_frames):
            target_ns = anchor_master_ns - offset * history_stride * interval_ns
            frame = self._nearest_master_frame(
                target_ns,
                robot_device=robot_device,
                anchor_ns=anchor_master_ns,
                minimum_ns=inference.episode_start_timestamp_ns,
            )
            if frame is not None:
                frame_master_ns = self._device_timestamp_ns(
                    frame.device_timestamps_ns,
                    robot_device,
                    frame.timestamp_ns,
                )
                if abs(frame_master_ns - target_ns) > 50_000_000:
                    frame = None
            if frame is None:
                samples.append(
                    {
                        name: np.array(value, copy=True)
                        for name, value in samples[-1].items()
                    }
                )
                timestamps.append(timestamps[-1])
                continue

            samples.append(
                self._policy_observation_from_values(
                    self._load_frame(frame),
                    camera_devices=camera_devices,
                    robot_device=robot_device,
                )
            )
            timestamps.append(frame.timestamp_ns)

        samples.reverse()
        timestamps.reverse()
        observation = {
            name: [np.array(sample[name], copy=True) for sample in samples]
            for name in (*camera_devices, robot_device)
        }
        return observation, timestamps, observation_timestamp_ns


def write_image(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
        path, format="JPEG", quality=92
    )


def observation_image_preview(
    observation: Mapping[str, Any],
    camera_devices: Sequence[str],
) -> np.ndarray:
    """Build a side-by-side RGB preview from the exact policy observation."""
    images: list[np.ndarray] = []
    for device in camera_devices:
        value = np.asarray(observation[device])
        while value.ndim > 3:
            value = value[-1]
        if value.ndim != 3:
            raise ValueError(f"Camera {device!r} must be HWC or CHW, got {value.shape}")
        if value.shape[0] in {1, 3, 4} and value.shape[-1] not in {1, 3, 4}:
            value = np.moveaxis(value, 0, -1)
        if value.shape[-1] == 1:
            value = np.repeat(value, 3, axis=-1)
        elif value.shape[-1] == 4:
            value = value[..., :3]
        if value.shape[-1] != 3:
            raise ValueError(f"Camera {device!r} has unsupported shape {value.shape}")
        value = np.asarray(value, dtype=np.float32)
        if (float(value.max()) if value.size else 0.0) <= 1.5:
            value *= 255.0
        images.append(np.clip(value, 0, 255).astype(np.uint8))
    if not images:
        raise ValueError("At least one camera is required for an observation preview")
    height = min(image.shape[0] for image in images)
    normalized = []
    from PIL import Image

    for image in images:
        width = max(1, round(image.shape[1] * height / image.shape[0]))
        normalized.append(np.asarray(Image.fromarray(image).resize((width, height))))
    return np.hstack(normalized)


def _validate_rigid_transform(transform: Any, name: str) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} transform must be 4x4, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} transform contains NaN or infinity")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} transform has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} transform rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} transform rotation determinant must be +1")
    return matrix


def _tcp_umi_transforms(
    tcp_umi_json_path: str | Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the source-compatible TCP->UMI and UMI->TCP transforms."""
    if tcp_umi_json_path in (None, ""):
        identity = np.eye(4, dtype=np.float64)
        return identity, identity

    calibration_path = Path(tcp_umi_json_path).expanduser()
    if not calibration_path.is_absolute():
        calibration_path = Path(__file__).resolve().parents[2] / calibration_path
    tcp_umi, umi_tcp = load_tcp_umi_transforms(str(calibration_path))
    tcp_umi = _validate_rigid_transform(tcp_umi, "tcp_umi")
    umi_tcp = _validate_rigid_transform(umi_tcp, "umi_tcp")
    if not np.allclose(tcp_umi @ umi_tcp, np.eye(4), atol=1e-8):
        raise ValueError("TCP/UMI calibration matrices are not inverses")
    return tcp_umi, umi_tcp


def _safe_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float64)
    if value.shape[-1:] != (4,):
        raise ValueError(f"Quaternion must have last dimension 4, got {value.shape}")
    if value.ndim == 1:
        norm = float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        normalized = value / norm
        return -normalized if normalized[3] < 0.0 else normalized

    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    valid = np.isfinite(norm) & (norm >= 1e-12)
    normalized = np.zeros_like(value, dtype=np.float64)
    normalized[..., 3] = 1.0
    normalized = np.where(valid, value / np.maximum(norm, 1e-12), normalized)
    return np.where(normalized[..., 3:4] < 0.0, -normalized, normalized)


def _pose_matrix(position: np.ndarray, rotation: Rotation) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return matrix


def _poses_to_matrices(
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    quaternions_xyzw = np.asarray(quaternions_xyzw, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Pose positions must be [T,3], got {positions.shape}")
    if quaternions_xyzw.shape != (positions.shape[0], 4):
        raise ValueError(
            "Pose quaternions must be [T,4] and match positions, got "
            f"{quaternions_xyzw.shape} vs {positions.shape}"
        )
    matrices = np.repeat(
        np.eye(4, dtype=np.float64)[None, :, :],
        positions.shape[0],
        axis=0,
    )
    matrices[:, :3, :3] = Rotation.from_quat(quaternions_xyzw).as_matrix()
    matrices[:, :3, 3] = positions
    return matrices


def tcp_state16_to_umi_state(
    tcp_state16: np.ndarray,
    *,
    tcp_umi_json_path: str | Path | None = None,
) -> np.ndarray:
    """Convert recorded TCP state `[xyz, wxyz, gripper] x 2` to UMI state."""
    state = np.asarray(tcp_state16, dtype=np.float64).reshape(-1)
    if state.size < 16:
        raise ValueError(f"Expected TCP state with at least 16 values, got {state.shape}")
    tcp_umi, _umi_tcp = _tcp_umi_transforms(tcp_umi_json_path)
    transformed_arms = []
    for arm_index in (0, 1):
        base = arm_index * 8
        anchor = state[base : base + 8]
        anchor_quat_xyzw = _safe_quat_xyzw(
            np.asarray([anchor[4], anchor[5], anchor[6], anchor[3]], dtype=np.float64)
        )
        umi_pose = _pose_matrix(
            anchor[:3],
            Rotation.from_quat(anchor_quat_xyzw),
        ) @ tcp_umi
        umi_quat_xyzw = _safe_quat_xyzw(
            Rotation.from_matrix(umi_pose[:3, :3]).as_quat()
        )
        transformed_arms.append(
            np.concatenate(
                [
                    umi_pose[:3, 3],
                    umi_quat_xyzw[3:4],
                    umi_quat_xyzw[:3],
                    anchor[7:8],
                ],
                axis=0,
            )
        )
    result = np.concatenate(transformed_arms, axis=0)
    if not np.all(np.isfinite(result)):
        raise ValueError("FastWAM UMI state contains NaN or infinity")
    return result


def relative_action_to_absolute(
    relative_action: np.ndarray,
    state16: np.ndarray,
    *,
    relative_pose_frame: str,
    action_quaternion_order: str,
    gripper_max_width: float | None = DEFAULT_GRIPPER_MAX_WIDTH,
    normalize_gripper_width: bool = False,
    clip_gripper_width: bool = True,
    tcp_umi_json_path: str | Path | None = None,
) -> np.ndarray:
    """Mirror FastWAMPolicy's UMI-relative to TCP-absolute conversion."""

    action = np.asarray(relative_action, dtype=np.float64)
    state = np.asarray(state16, dtype=np.float64).reshape(-1)
    if action.ndim != 2 or action.shape[1] < 8 or state.size < 16:
        return action
    tcp_umi, umi_tcp = _tcp_umi_transforms(tcp_umi_json_path)
    order = action_quaternion_order.lower()
    if order not in {"xyzw", "wxyz"}:
        raise ValueError(f"Unsupported FastWAM action quaternion order: {order!r}")
    result = action.copy()
    for arm_index in range(2 if action.shape[1] >= 16 else 1):
        base = arm_index * 8
        anchor = state[base : base + 8]
        anchor_quat_xyzw = _safe_quat_xyzw(np.asarray(
            [anchor[4], anchor[5], anchor[6], anchor[3]], dtype=np.float64
        ))
        anchor_tcp = _pose_matrix(anchor[:3], Rotation.from_quat(anchor_quat_xyzw))
        anchor_umi = anchor_tcp @ tcp_umi
        rel_quat = action[:, base + 3 : base + 7]
        if order == "wxyz":
            rel_quat = rel_quat[:, [1, 2, 3, 0]]
        rel_quat = _safe_quat_xyzw(rel_quat)
        rel_pos = action[:, base : base + 3]
        relative_frame = relative_pose_frame.lower()
        if relative_frame == "local_frame":
            target_umi = anchor_umi @ _poses_to_matrices(rel_pos, rel_quat)
        elif relative_frame == "world_frame":
            target_umi = np.repeat(
                np.eye(4, dtype=np.float64)[None, :, :],
                rel_pos.shape[0],
                axis=0,
            )
            target_umi[:, :3, :3] = (
                anchor_umi[:3, :3] @ Rotation.from_quat(rel_quat).as_matrix()
            )
            target_umi[:, :3, 3] = anchor_umi[None, :3, 3] + rel_pos
        else:
            raise ValueError(
                f"Invalid FastWAM relative_pose_frame: {relative_pose_frame!r}"
            )
        target_tcp = target_umi @ umi_tcp
        result[:, base : base + 3] = target_tcp[:, :3, 3]
        result[:, base + 3 : base + 7] = _safe_quat_xyzw(
            Rotation.from_matrix(target_tcp[:, :3, :3]).as_quat()
        )
    if gripper_max_width is not None:
        max_width = float(gripper_max_width)
        if max_width <= 0:
            raise ValueError("gripper_max_width must be positive")
        for index in (7, 15):
            if index < result.shape[1]:
                if clip_gripper_width or normalize_gripper_width:
                    result[:, index] = np.clip(result[:, index], 0.0, max_width)
                if normalize_gripper_width:
                    result[:, index] /= max_width
    return result


def _tensor_to_rgb(tensor: Any) -> np.ndarray:
    value = tensor.detach().float().cpu().clamp(-1.0, 1.0)
    if value.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(value.shape)}")
    if value.shape[0] >= 6:
        return np.hstack([_tensor_to_rgb(value[:3]), _tensor_to_rgb(value[3:6])])
    if value.shape[0] == 1:
        value = value.repeat(3, 1, 1)
    return (
        value[:3].permute(1, 2, 0).numpy() * 127.5 + 127.5
    ).round().astype(np.uint8)


def input_tensor_preview(
    input_image: Any,
    input_video: Any,
) -> np.ndarray:
    if input_video is not None:
        value = input_video[0] if input_video.ndim == 5 else input_video
        if value.ndim != 4:
            raise ValueError(f"Expected CTHW video, got {tuple(value.shape)}")
        frames = [_tensor_to_rgb(value[:, index]) for index in range(value.shape[1])]
        while len(frames) < 4:
            frames.append(frames[-1].copy())
        return np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:4])])
    if input_image is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    value = input_image[0] if input_image.ndim == 4 else input_image
    return _tensor_to_rgb(value)
