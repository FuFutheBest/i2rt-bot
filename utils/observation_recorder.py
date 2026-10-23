"""Timestamped observation recording for reproducible FastWAM comparisons.

The recorder keeps two related streams:

* ``frames.jsonl`` + ``frames/*.npz`` contain the raw, synchronized device
  observations sampled from summary SHM.  These frames let a later checkpoint
  reconstruct a different visual-history stride without touching the robot.
* ``inference.jsonl`` + ``inference/*.npz`` contain the exact policy input
  assembled for each real inference, including the history timestamps selected
  by ``PolicyConnector``.  They are the audit record for what the model saw.

The writer is asynchronous so JPEG/NPZ compression does not extend the policy
control loop.  ``stop()`` drains the queue before returning.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from utils.inference_options import as_bool
from utils.logger_config import logger
from utils.superoffline_observation import (
    RECORDER_OBSERVATION_CONTRACT,
    ObservationContractError,
    device_roles_from_arrays,
    reviewed_calibration_files,
    source_profile_provenance_from_files,
    validate_capture_device_identity,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _timestamp_label(timestamp_ns: int) -> str:
    seconds, nanos = divmod(int(timestamp_ns), 1_000_000_000)
    local = time.localtime(seconds)
    return time.strftime("%Y%m%d_%H%M%S", local) + f"_{nanos:09d}"


def _calibration_files_from_config(raw_config: Mapping[str, Any]) -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    expected = reviewed_calibration_files(repo_root)
    raw_files = raw_config.get("calibration_files")
    if raw_files is None:
        return expected
    if not isinstance(raw_files, Mapping):
        raise ValueError("observation_recorder.calibration_files must be an object")
    files: dict[str, Path] = {}
    for raw_label, raw_path in raw_files.items():
        label = str(raw_label).strip()
        path_text = str(raw_path or "").strip()
        if not label or not path_text:
            raise ValueError(
                "observation_recorder.calibration_files must map non-empty labels to paths"
            )
        path = Path(path_text).expanduser()
        files[label] = path if path.is_absolute() else repo_root / path
    if set(files) != set(expected):
        raise ValueError(
            "observation_recorder.calibration_files must declare the reviewed bundle"
        )
    for label, expected_path in expected.items():
        if files[label].resolve() != expected_path.resolve():
            raise ValueError(
                "observation_recorder.calibration_files must use the reviewed bundle"
            )
    return files


@dataclass
class _Record:
    kind: str
    index: int
    timestamp_ns: int
    arrays: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


class TimestampedObservationRecorder:
    """Persist raw summary observations and exact inference inputs."""

    schema_version = 2

    def __init__(
        self,
        config: Optional[Mapping[str, Any]],
        *,
        base_dir: Optional[Path] = None,
        run_started_at_ns: Optional[int] = None,
        device_names: Optional[list[str]] = None,
        camera_devices: Optional[list[str]] = None,
        robot_device: Optional[str] = None,
    ) -> None:
        raw_config = dict(config or {})
        self.enabled = as_bool(raw_config.get("enabled", False))
        raw_source_profile = raw_config.get("superoffline_reviewed_source_profile", raw_config.get("source_profile"))
        self.source_profile = str(raw_source_profile or "").strip()
        if self.enabled and not self.source_profile:
            raise ValueError(
                "observation_recorder.superoffline_reviewed_source_profile is required when recording is enabled"
            )
        self.device_names = list(device_names or [])
        self.camera_devices = list(camera_devices or [])
        self.robot_device = robot_device
        self.source_provenance: Dict[str, Any] | None = None
        if self.enabled:
            validate_capture_device_identity(
                self.source_profile,
                camera_devices=self.camera_devices,
                robot_device=str(self.robot_device or ""),
            )
            capture_device_descriptors = raw_config.get("capture_device_descriptors")
            if capture_device_descriptors is None:
                raise ValueError(
                    "observation_recorder.capture_device_descriptors must be derived "
                    "from the composed device configuration"
                )
            self.source_provenance = source_profile_provenance_from_files(
                self.source_profile,
                calibration_files=_calibration_files_from_config(raw_config),
                capture_device_descriptors=capture_device_descriptors,
            )
        self.record_fps = float(raw_config.get("fps", 30.0))
        if self.record_fps <= 0:
            raise ValueError("observation_recorder.fps must be positive")
        self.queue_size = int(raw_config.get("queue_size", 512))
        if self.queue_size <= 0:
            raise ValueError("observation_recorder.queue_size must be positive")
        self.compress = as_bool(raw_config.get("compress", False))
        self.base_dir = Path(
            raw_config.get("output_dir", base_dir or "logs/fastwam_observations")
        ).expanduser()
        self.run_started_at_ns = int(run_started_at_ns or time.time_ns())
        self.run_dir = self.base_dir / f"run_{_timestamp_label(self.run_started_at_ns)}"
        self.frames_dir = self.run_dir / "frames"
        self.inference_dir = self.run_dir / "inference"
        self.manifest_path = self.run_dir / "manifest.json"
        self.frames_index_path = self.run_dir / "frames.jsonl"
        self.inference_index_path = self.run_dir / "inference.jsonl"
        self._frame_index = 0
        self._inference_index = 0
        self._last_sample_ns: Optional[int] = None
        self._queue: queue.Queue[Optional[_Record]] = queue.Queue(
            maxsize=self.queue_size
        )
        self._writer_thread: Optional[threading.Thread] = None
        self._sampling_thread: Optional[threading.Thread] = None
        self._sampling_stop = threading.Event()
        self._writer_stop = threading.Event()
        self._read_callback: Optional[Callable[[], Any]] = None
        self._lock = threading.Lock()
        self._dropped_frames = 0
        self._written_frames = 0
        self._written_inferences = 0
        self._write_failures = 0
        self._episode_lock = threading.Lock()
        self._manifest_role_lock = threading.Lock()
        self._device_roles: Dict[str, Any] | None = None

    def _ensure_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.inference_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            manifest = {
                "schema": "superinference.timestamped_observation",
                "schema_version": self.schema_version,
                "run_started_at_ns": self.run_started_at_ns,
                "run_started_at_local": _timestamp_label(self.run_started_at_ns),
                "record_fps": self.record_fps,
                "npz_compressed": self.compress,
                "device_names": self.device_names,
                "camera_devices": self.camera_devices,
                "robot_device": self.robot_device,
                "observation_contract": RECORDER_OBSERVATION_CONTRACT,
                "source_profile": self.source_profile,
                "source_provenance": self.source_provenance,
                "frames_index": self.frames_index_path.name,
                "inference_index": self.inference_index_path.name,
                "notes": (
                    "frames are raw synchronized summary-SHM observations; "
                    "inference records are exact policy inputs"
                ),
            }
            self.manifest_path.write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

    def _publish_device_roles(self, arrays: Mapping[str, np.ndarray]) -> None:
        """Freeze role names, dtype, and shape from the first raw frame."""
        if self._device_roles is not None:
            candidate = device_roles_from_arrays(
                camera_devices=self.camera_devices,
                robot_device=str(self.robot_device or ""),
                arrays=arrays,
            )
            if candidate != self._device_roles:
                raise ObservationContractError(
                    "recorder device role dtype/shape changed during one run"
                )
            return
        with self._manifest_role_lock:
            candidate = device_roles_from_arrays(
                camera_devices=self.camera_devices,
                robot_device=str(self.robot_device or ""),
                arrays=arrays,
            )
            if self._device_roles is not None:
                if candidate != self._device_roles:
                    raise ObservationContractError(
                        "recorder device role dtype/shape changed during one run"
                    )
                return
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            manifest["device_roles"] = candidate
            self.manifest_path.write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            self._device_roles = candidate

    def start(self, read_callback: Callable[[], Any]) -> Optional[Path]:
        """Start asynchronous raw-frame sampling and return the run directory."""
        if not self.enabled:
            return None
        if self._sampling_thread is not None:
            return self.run_dir
        self._ensure_dirs()
        self._read_callback = read_callback
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="TimestampedObservationWriter",
            daemon=True,
        )
        self._writer_thread.start()
        self._sampling_thread = threading.Thread(
            target=self._sampling_loop,
            name="TimestampedObservationSampler",
            daemon=True,
        )
        self._sampling_thread.start()
        logger.info(
            "Timestamped observation recording enabled at {} ({:.1f} Hz)",
            self.run_dir.resolve(),
            self.record_fps,
        )
        return self.run_dir

    def mark_episode(
        self,
        episode_index: int,
        episode_start_timestamp_ns: int,
    ) -> None:
        """Append a policy episode boundary to the run-level audit log."""
        if not self.enabled:
            return
        self._ensure_dirs()
        row = {
            "episode_index": int(episode_index),
            "episode_start_timestamp_ns": int(episode_start_timestamp_ns),
            "episode_start_local": _timestamp_label(episode_start_timestamp_ns),
        }
        with self._episode_lock:
            with (self.run_dir / "episodes.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(row, separators=(",", ":")) + "\n")

    def _sampling_loop(self) -> None:
        interval_s = 1.0 / self.record_fps
        next_deadline = time.monotonic()
        while not self._sampling_stop.is_set():
            if self._read_callback is not None:
                try:
                    result = self._read_callback()
                    if result is not None:
                        frame_timestamp_ns, device_data = result
                        self.record_frame(
                            int(frame_timestamp_ns),
                            device_data,
                        )
                except Exception as exc:
                    logger.debug("Observation sampler skipped frame: {}", exc)
            next_deadline += interval_s
            wait_s = max(0.0, next_deadline - time.monotonic())
            self._sampling_stop.wait(wait_s)

    @staticmethod
    def _normalize_device_data(
        device_data: Mapping[str, Any],
    ) -> tuple[Dict[str, np.ndarray], Dict[str, int]]:
        arrays: Dict[str, np.ndarray] = {}
        timestamps: Dict[str, int] = {}
        for device_name, item in device_data.items():
            if isinstance(item, tuple) and len(item) == 2:
                timestamp_ns, data = item
            else:
                timestamp_ns, data = 0, item
            arrays[str(device_name)] = np.array(data, copy=True)
            try:
                timestamps[str(device_name)] = int(timestamp_ns)
            except (TypeError, ValueError, OverflowError):
                timestamps[str(device_name)] = 0
        return arrays, timestamps

    @staticmethod
    def _array_key(prefix: str, device_name: str) -> str:
        """Encode a device key injectively for an NPZ member name."""
        return f"{prefix}__{device_name.encode('utf-8').hex()}"

    def _enqueue(self, record: _Record) -> None:
        try:
            self._queue.put(record, timeout=0.25)
        except queue.Full:
            if record.kind == "frame":
                self._dropped_frames += 1
                logger.warning(
                    "Observation recorder queue is full; dropped raw frame #{}",
                    record.index,
                )
            else:
                # Inference records are the audit trail; block briefly rather
                # than silently losing the exact model input.
                self._queue.put(record)

    def record_frame(
        self,
        summary_timestamp_ns: int,
        device_data: Mapping[str, Any],
    ) -> None:
        if not self.enabled:
            return
        timestamp_ns = int(summary_timestamp_ns)
        if timestamp_ns <= 0:
            return
        # Avoid writing the same summary slot repeatedly when sampling faster
        # than the device manager updates it.
        with self._lock:
            if self._last_sample_ns == timestamp_ns:
                return
            self._last_sample_ns = timestamp_ns
            index = self._frame_index
            self._frame_index += 1
        arrays, device_timestamps = self._normalize_device_data(device_data)
        self._publish_device_roles(arrays)
        array_keys: Dict[str, str] = {}
        encoded_arrays: Dict[str, np.ndarray] = {}
        for device_name, array in arrays.items():
            array_key = self._array_key("device", device_name)
            if array_key in encoded_arrays:
                raise ValueError(
                    f"observation recorder array-key collision for {device_name!r}"
                )
            encoded_arrays[array_key] = array
            array_keys[device_name] = array_key
        metadata = {
            "summary_timestamp_ns": timestamp_ns,
            "summary_timestamp_local": _timestamp_label(timestamp_ns),
            "device_timestamps_ns": device_timestamps,
            "device_names": sorted(arrays),
            "device_array_keys": array_keys,
            "dropped_frames_before_this_record": self._dropped_frames,
        }
        self._enqueue(_Record("frame", index, timestamp_ns, encoded_arrays, metadata))

    def record_inference(
        self,
        observation_timestamp_ns: int,
        policy_input: Mapping[str, Any],
        *,
        action_index: Optional[int] = None,
        inference_start_timestamp_ns: Optional[int] = None,
        summary_timestamp_ns: Optional[int] = None,
        device_timestamps_ns: Optional[Mapping[str, int]] = None,
        episode_index: Optional[int] = None,
        episode_start_timestamp_ns: Optional[int] = None,
    ) -> None:
        """Record the exact input dictionary passed to ``FastWAMPolicy``."""
        if not self.enabled:
            return
        timestamp_ns = int(observation_timestamp_ns)
        if timestamp_ns <= 0:
            return
        with self._lock:
            index = self._inference_index
            self._inference_index += 1

        arrays: Dict[str, np.ndarray] = {}
        metadata: Dict[str, Any] = {
            "observation_timestamp_ns": timestamp_ns,
            "observation_timestamp_local": _timestamp_label(timestamp_ns),
            "action_index": action_index,
            "inference_start_timestamp_ns": inference_start_timestamp_ns,
            "summary_timestamp_ns": summary_timestamp_ns,
            "episode_index": episode_index,
            "episode_start_timestamp_ns": episode_start_timestamp_ns,
            "device_timestamps_ns": {
                str(key): int(value)
                for key, value in dict(device_timestamps_ns or {}).items()
            },
            "keys": [],
        }
        for key, value in policy_input.items():
            if str(key).endswith("_is_pad"):
                metadata[str(key)] = _json_safe(value)
                continue
            if str(key) == "_history_frame_timestamps_ns":
                metadata[str(key)] = _json_safe(value)
                continue
            try:
                array = np.asarray(value)
            except Exception:
                metadata[str(key)] = _json_safe(value)
                continue
            if array.dtype.kind in {"O", "U", "S"}:
                metadata[str(key)] = _json_safe(value)
                continue
            array_key = self._array_key("obs", str(key))
            if array_key in arrays:
                raise ValueError(
                    f"observation recorder array-key collision for {key!r}"
                )
            arrays[array_key] = np.array(array, copy=True)
            metadata["keys"].append(
                {"name": str(key), "array_key": array_key, "shape": list(array.shape)}
            )
        self._enqueue(_Record("inference", index, timestamp_ns, arrays, metadata))

    def _writer_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                if self._writer_stop.is_set():
                    return
                continue
            if item is None:
                self._queue.task_done()
                return
            try:
                self._write_record(item)
            except Exception as exc:
                self._write_failures += 1
                logger.warning(
                    "Failed to write {} observation record: {}", item.kind, exc
                )
            finally:
                self._queue.task_done()

    def _write_record(self, record: _Record) -> None:
        if record.kind == "frame":
            relative_dir = self.frames_dir
            index_path = self.frames_index_path
        else:
            relative_dir = self.inference_dir
            index_path = self.inference_index_path
        npz_name = f"{record.index:08d}_{_timestamp_label(record.timestamp_ns)}.npz"
        npz_path = relative_dir / npz_name
        writer = np.savez_compressed if self.compress else np.savez
        writer(npz_path, **record.arrays)
        line = dict(record.metadata)
        line.update(
            {
                "schema_version": self.schema_version,
                "kind": record.kind,
                "index": record.index,
                "timestamp_ns": record.timestamp_ns,
                "file": str(npz_path.relative_to(self.run_dir)),
            }
        )
        with index_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(_json_safe(line), separators=(",", ":")) + "\n")
        if record.kind == "frame":
            self._written_frames += 1
        else:
            self._written_inferences += 1

    def stop(self) -> None:
        if not self.enabled:
            return
        self._sampling_stop.set()
        if self._sampling_thread is not None:
            self._sampling_thread.join(timeout=3.0)
            self._sampling_thread = None
        self._writer_stop.set()
        self._queue.join()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=3.0)
            self._writer_thread = None
        try:
            self.manifest_path.write_text(
                json.dumps(
                    {
                        **json.loads(self.manifest_path.read_text(encoding="utf-8")),
                        "stopped_at_ns": time.time_ns(),
                        "frame_count": self._written_frames,
                        "inference_count": self._written_inferences,
                        "attempted_frame_count": self._frame_index,
                        "attempted_inference_count": self._inference_index,
                        "dropped_frames": self._dropped_frames,
                        "write_failures": self._write_failures,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to finalize observation recorder manifest: {}", exc)
        else:
            logger.info(
                "Timestamped observation recording finalized at {}: "
                "frames={}, inferences={}, dropped_frames={}, write_failures={}",
                self.run_dir.resolve(),
                self._written_frames,
                self._written_inferences,
                self._dropped_frames,
                self._write_failures,
            )
