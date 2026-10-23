#!/usr/bin/env python3
"""Run the I2RT NEDF3 terminal pipeline with simulated shared-memory devices."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from consumer.io.async_data_saver_i2rt_NEDF3 import (  # noqa: E402
    I2RTNEDF3Saver,
    SaverConfig,
)
from consumer.io.nedf3_i2rt_contract import (  # noqa: E402
    NEDF3_CAMERA_TOPICS,
    NEDF3_I2RT_JOINT_TOPICS,
    NEDF3_RECORD_TOPICS,
)
from devices.base import BaseDevice  # noqa: E402


def _publisher(name: str, shape: tuple[int, ...], dtype: str) -> BaseDevice:
    dimension = int(np.prod(shape))
    device = BaseDevice(
        device_id=0,
        device_name=name,
        state_feedback={
            "fps": 30.0,
            "buffer_size": 8,
            "hardware_latency_ms": 0.0,
            "format": {
                "shape": list(shape),
                "dtype": dtype,
                "type": [f"linear:{dimension}"],
            },
        },
    )
    device._create_shared_memory()
    return device


def run_unreal_smoke(output_dir: Path, *, frames: int = 4) -> Path:
    """Write and reopen a complete NEDF3 episode without robot hardware."""
    if frames <= 0:
        raise ValueError("frames must be positive")
    suffix = uuid.uuid4().hex[:10]
    names = {
        role: f"I2RTUnreal_{role}_{suffix}"
        for role in ("leader", "follower", "right", "left")
    }
    publishers = {
        "leader": _publisher(names["leader"], (14,), "float64"),
        "follower": _publisher(names["follower"], (14,), "float64"),
        "right": _publisher(names["right"], (360, 640, 3), "uint8"),
        "left": _publisher(names["left"], (360, 640, 3), "uint8"),
    }
    config = SaverConfig(
        output_dir=output_dir,
        sample_hz=25.0,
        queue_size=64,
        jpeg_quality=85,
        leader_shm=f"{names['leader']}_0_data",
        follower_shm=f"{names['follower']}_0_data",
        camera_shm={
            "right": f"{names['right']}_0_data",
            "left": f"{names['left']}_0_data",
            "head": None,
        },
        task="unreal_smoke",
        task_description="hardware-free I2RT NEDF3 pipeline verification",
        synthetic_camera_shapes={"head": (480, 640, 3)},
    )
    saver = I2RTNEDF3Saver(config)
    try:
        saver.connect()
        saver.start_episode()
        initial_timestamp_ns = time.time_ns()
        for frame_index in range(frames):
            timestamp_ns = initial_timestamp_ns + frame_index * 40_000_000
            leader = np.arange(14, dtype=np.float64) + frame_index * 0.01
            follower = leader + 0.5
            publishers["leader"]._write_array_to_shm_with_timestamp(
                leader, timestamp_ns
            )
            publishers["follower"]._write_array_to_shm_with_timestamp(
                follower, timestamp_ns + 1
            )
            for camera_index, role in enumerate(("right", "left")):
                image = np.full(
                    (360, 640, 3),
                    30 + frame_index * 10 + camera_index,
                    dtype=np.uint8,
                )
                publishers[role]._write_array_to_shm_with_timestamp(
                    image, timestamp_ns + 2 + camera_index
                )
            saver.collect_once()
        episode_dir = saver.stop_episode()
    finally:
        saver.close()
        for publisher in publishers.values():
            publisher._cleanup_shared_memory()

    if episode_dir.parent.name == "quarantine":
        raise RuntimeError(f"unreal episode was quarantined: {episode_dir}")

    metadata = json.loads((episode_dir / "metadata.json").read_text())
    if metadata.get("main_camera") != "camera_0":
        raise RuntimeError(
            "two-camera episode did not select the real right wrist as main_camera"
        )

    from dm3_core.nedf import NEDFFactory, NEDFReaderConfig

    reader = NEDFFactory.get_reader(
        NEDFReaderConfig(metadata_file_path=episode_dir / "metadata.json")
    )
    if not reader.check_valid():
        raise RuntimeError("unreal NEDF3 episode failed reader validation")
    for topic in NEDF3_RECORD_TOPICS:
        messages = list(reader.iter_decoded_messages(topic))
        if not messages:
            raise RuntimeError(f"unreal NEDF3 episode has no messages for {topic}")

    left_topic = NEDF3_I2RT_JOINT_TOPICS["leader"]["left"]
    first_left = next(reader.iter_decoded_messages(left_topic)).decoded_message
    positions = np.asarray([joint.pos for joint in first_left.joints])
    np.testing.assert_allclose(positions, np.arange(6, dtype=np.float64))
    expected_camera_sizes = {
        "right": (640, 480),
        "left": (640, 480),
        "head": (480, 640),
    }
    for role, expected_size in expected_camera_sizes.items():
        message = next(
            reader.iter_decoded_messages(NEDF3_CAMERA_TOPICS[role])
        ).decoded_message
        assert (message.rgb.rows, message.rgb.cols) == expected_size
    return episode_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frames", type=int, default=4)
    args = parser.parse_args(argv)
    output_dir = args.output_dir or Path(
        tempfile.mkdtemp(prefix="superinference_i2rt_nedf3_unreal_")
    )
    episode_dir = run_unreal_smoke(output_dir, frames=args.frames)
    print(f"Unreal I2RT NEDF3 smoke test passed: {episode_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
