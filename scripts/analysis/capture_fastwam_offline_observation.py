#!/usr/bin/env python3
"""Capture one read-only FastWAM observation from device shared memory."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualizers.base_visualizer import BaseVisualizer  # noqa: E402
from utils.superoffline_observation import (  # noqa: E402
    RECORDER_OBSERVATION_CONTRACT,
    capture_device_descriptors_from_config,
    device_roles_from_arrays,
    reviewed_calibration_files,
    source_profile_provenance_from_files,
    validate_capture_device_identity,
)


def _read_recent(
    shm_name: str,
    dtype: Any,
    count: int,
) -> list[tuple[int, np.ndarray]]:
    reader = BaseVisualizer(shared_memory_name=shm_name, data_dtype=dtype)
    try:
        if not reader.connect():
            raise RuntimeError(f"Could not connect to shared memory {shm_name}")
        buffer_info = reader.get_buffer_info()
        if buffer_info is None:
            raise RuntimeError(f"Could not read buffer metadata from {shm_name}")
        available = min(int(count), int(buffer_info["frames_available"]))
        frames: list[tuple[int, np.ndarray]] = []
        for index in range(-available, 0):
            timestamp_ns, value = reader.read_frame_at_index(index)
            if timestamp_ns is not None and value is not None:
                frames.append((int(timestamp_ns), np.array(value, copy=True)))
                del value
        if not frames:
            raise RuntimeError(f"No frame is available in shared memory {shm_name}")
        frames.sort(key=lambda item: item[0])
        return frames
    finally:
        reader.disconnect()
        plt.close(reader.fig)


def _nearest(
    frames: list[tuple[int, np.ndarray]],
    target_timestamp_ns: int,
) -> tuple[int, np.ndarray]:
    return min(frames, key=lambda item: abs(item[0] - target_timestamp_ns))


def _validate_trusted_capture_shm_names(
    *,
    left_device: str,
    right_device: str,
    robot_device: str,
    left_shm: str,
    right_shm: str,
    robot_shm: str,
) -> None:
    """Reject SHM aliases that could mislabel an unrelated physical capture."""
    expected = {
        "--left-shm": f"{left_device}_data",
        "--right-shm": f"{right_device}_data",
        "--robot-shm": f"{robot_device}_data",
    }
    supplied = {
        "--left-shm": left_shm,
        "--right-shm": right_shm,
        "--robot-shm": robot_shm,
    }
    mismatches = [
        f"{argument} must equal {expected_name!r}, got {supplied[argument]!r}"
        for argument, expected_name in expected.items()
        if supplied[argument] != expected_name
    ]
    if mismatches:
        raise ValueError(
            "trusted SuperOffline capture reads only the selected devices' data SHM: "
            + "; ".join(mismatches)
        )


def write_observation_run(
    output_dir: Path,
    *,
    arrays: dict[str, np.ndarray],
    device_timestamps_ns: dict[str, int],
    camera_devices: list[str],
    robot_device: str,
    source_profile: str = "rizon-umi3-fisheye-openpi-rgb.v1",
    calibration_files: Mapping[str, str | Path] | None = None,
    capture_device_descriptors: Mapping[str, Any] | None = None,
    frame_samples: list[tuple[int, dict[str, np.ndarray], dict[str, int]]]
    | None = None,
) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    frames_dir = output_dir / "frames"
    inference_dir = output_dir / "inference"
    frames_dir.mkdir()
    inference_dir.mkdir()
    resolved_calibration_files = dict(
        reviewed_calibration_files(REPO_ROOT)
        if calibration_files is None
        else calibration_files
    )
    validate_capture_device_identity(
        source_profile,
        camera_devices=camera_devices,
        robot_device=robot_device,
    )
    if capture_device_descriptors is None:
        raise ValueError(
            "capture_device_descriptors must come from the composed device configuration"
        )

    timestamp_ns = max(device_timestamps_ns.values())
    relative_inference = Path("inference") / "inference_000000.npz"
    np.savez_compressed(output_dir / relative_inference, **arrays)

    if frame_samples is None:
        frame_samples = [(timestamp_ns, arrays, device_timestamps_ns)]
    frame_records = []
    for index, (frame_timestamp_ns, frame_arrays, frame_device_timestamps) in enumerate(
        frame_samples
    ):
        relative_frame = Path("frames") / f"frame_{index:06d}.npz"
        np.savez_compressed(output_dir / relative_frame, **frame_arrays)
        frame_records.append(
            {
                "kind": "frame",
                "index": index,
                "timestamp_ns": frame_timestamp_ns,
                "summary_timestamp_ns": frame_timestamp_ns,
                "file": relative_frame.as_posix(),
                "device_timestamps_ns": frame_device_timestamps,
            }
        )

    manifest = {
        "schema": "superinference.timestamped_observation",
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "source": "read_only_device_shm",
        "camera_devices": camera_devices,
        "robot_device": robot_device,
        "observation_contract": RECORDER_OBSERVATION_CONTRACT,
        "source_profile": source_profile,
        "source_provenance": source_profile_provenance_from_files(
            source_profile,
            calibration_files=resolved_calibration_files,
            capture_device_descriptors=capture_device_descriptors,
        ),
        "device_roles": device_roles_from_arrays(
            camera_devices=camera_devices,
            robot_device=robot_device,
            arrays=arrays,
        ),
        "frame_count": len(frame_records),
        "inference_count": 1,
    }
    inference_record = {
        "kind": "inference",
        "index": 0,
        "timestamp_ns": timestamp_ns,
        "observation_timestamp_ns": timestamp_ns,
        "observation_timestamp_local": datetime.fromtimestamp(
            timestamp_ns / 1e9
        ).astimezone().isoformat(),
        "summary_timestamp_ns": timestamp_ns,
        "file": relative_inference.as_posix(),
        "device_timestamps_ns": device_timestamps_ns,
        "keys": [
            {"name": name, "array_key": name, "shape": list(value.shape)}
            for name, value in arrays.items()
        ],
    }
    (output_dir / "frames.jsonl").write_text(
        "".join(
            json.dumps(record, separators=(",", ":")) + "\n"
            for record in frame_records
        ),
        encoding="utf-8",
    )
    (output_dir / "inference.jsonl").write_text(
        json.dumps(inference_record, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest["stopped_at_ns"] = time.time_ns()
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-shm", default=None)
    parser.add_argument("--right-shm", default=None)
    parser.add_argument("--robot-shm", default=None)
    parser.add_argument("--left-device", default=None)
    parser.add_argument("--right-device", default=None)
    parser.add_argument("--robot-device", default=None)
    parser.add_argument(
        "--source-profile",
        default=None,
        help="Reviewed recorder source profile used for SuperOffline compatibility.",
    )
    parser.add_argument(
        "--config-name",
        default="policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam",
        help=(
            "Composed real-robot launch config used to attest declared camera rotation "
            "and robot serial identity."
        ),
    )
    parser.add_argument(
        "--calibration-file",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Override one reviewed calibration file recorded in the source provenance.",
    )
    parser.add_argument(
        "--buffer-frames",
        type=int,
        default=32,
        help="Recent frames to read from each observation SHM buffer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New observation run directory. Defaults below logs/offline_observations.",
    )
    args = parser.parse_args()

    from incubator.control_panel.runtime_config import compose_runtime_config

    composed_config = compose_runtime_config(args.config_name)
    observations = composed_config.get("policy", {}).get("obs", {}).get("devices", [])
    if len(observations) != 3:
        parser.error("capture requires ordered policy observations: left camera, right camera, robot")
    args.left_device = args.left_device or observations[0]
    args.right_device = args.right_device or observations[1]
    args.robot_device = args.robot_device or observations[2]
    args.left_shm = args.left_shm or f"{args.left_device}_data"
    args.right_shm = args.right_shm or f"{args.right_device}_data"
    args.robot_shm = args.robot_shm or f"{args.robot_device}_data"
    args.source_profile = args.source_profile or composed_config.get("policy", {}).get(
        "observation_recorder", {}
    ).get("superoffline_reviewed_source_profile")
    if not args.source_profile:
        parser.error("configure observation_recorder.superoffline_reviewed_source_profile or --source-profile")

    try:
        _validate_trusted_capture_shm_names(
            left_device=args.left_device,
            right_device=args.right_device,
            robot_device=args.robot_device,
            left_shm=args.left_shm,
            right_shm=args.right_shm,
            robot_shm=args.robot_shm,
        )
    except ValueError as exc:
        parser.error(str(exc))

    calibration_files = reviewed_calibration_files(REPO_ROOT)
    for item in args.calibration_file:
        label, separator, raw_path = str(item).partition("=")
        if not separator or not label.strip() or not raw_path.strip():
            parser.error("--calibration-file must use LABEL=PATH")
        calibration_files[label.strip()] = Path(raw_path.strip()).expanduser()

    capture_device_descriptors = capture_device_descriptors_from_config(
        args.source_profile,
        devices=composed_config["devices"],
        camera_devices=[args.left_device, args.right_device],
        robot_device=args.robot_device,
    )

    output_dir = args.output_dir
    if output_dir is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("logs/offline_observations") / run_id
    output_dir = output_dir.expanduser().resolve()

    left_history = _read_recent(args.left_shm, np.uint8, args.buffer_frames)
    right_history = _read_recent(args.right_shm, np.uint8, args.buffer_frames)
    robot_history = _read_recent(args.robot_shm, np.float64, args.buffer_frames)
    left_ts, left = left_history[-1]
    right_ts, right = right_history[-1]
    robot_ts, robot = robot_history[-1]
    if robot.reshape(-1).size < 44:
        raise ValueError(f"Robot observation must contain at least 44 values: {robot.shape}")

    frame_samples = []
    for frame_robot_ts, frame_robot in robot_history:
        frame_left_ts, frame_left = _nearest(left_history, frame_robot_ts)
        frame_right_ts, frame_right = _nearest(right_history, frame_robot_ts)
        timestamps = {
            args.left_device: frame_left_ts,
            args.right_device: frame_right_ts,
            args.robot_device: frame_robot_ts,
        }
        frame_samples.append(
            (
                max(timestamps.values()),
                {
                    args.left_device: frame_left,
                    args.right_device: frame_right,
                    args.robot_device: frame_robot,
                },
                timestamps,
            )
        )

    result = write_observation_run(
        output_dir,
        arrays={
            args.left_device: left,
            args.right_device: right,
            args.robot_device: robot,
        },
        device_timestamps_ns={
            args.left_device: left_ts,
            args.right_device: right_ts,
            args.robot_device: robot_ts,
        },
        camera_devices=[args.left_device, args.right_device],
        robot_device=args.robot_device,
        source_profile=args.source_profile,
        calibration_files=calibration_files,
        capture_device_descriptors=capture_device_descriptors,
        frame_samples=frame_samples,
    )
    print(result)


if __name__ == "__main__":
    main()
