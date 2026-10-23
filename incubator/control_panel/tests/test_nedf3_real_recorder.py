from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid

import numpy as np
import pytest


pytest.importorskip("dm3_core.nedf")
pytest.importorskip("nmx_msg")

from incubator.r1.async_data_saver_NEDF3_r1 import R1DataSaverConsumer as DataSaverConsumer
from incubator.control_panel.nedf3_io import (
    extract_gripper_width,
    extract_joint_state_positions,
    extract_pose_wxyz,
)
from incubator.control_panel.recording_contract import (
    NEDF3_ARM_TOPICS,
    NEDF3_CAMERA_IDS,
    NEDF3_DATA_FORMAT,
    NEDF3_HAND_TOPICS,
    NEDF3_HEAD_TOPICS,
    NEDF3_RECORD_TOPIC_LIST,
)
from incubator.control_panel.recording_control import write_conversion_source_receipt
from dm3_core.nedf import NEDFFactory, NEDFReaderConfig


FRAME_COUNT = 8


def _task(
    device_key: str,
    device_type: str,
    timestamp_ns: int,
    data: np.ndarray,
) -> dict[str, object]:
    return {
        "device_key": device_key,
        "device_type": device_type,
        "device_name": device_key,
        "timestamp_ns": timestamp_ns,
        "data_array": data,
        "device_info": {"type": device_type, "shape": list(data.shape)},
    }


def _rizon_state(frame_index: int) -> np.ndarray:
    values = np.zeros(44, dtype=np.float32)
    values[0:7] = np.linspace(0.01, 0.07, 7) + frame_index / 1000.0
    values[14:21] = [
        0.40 + frame_index / 1000.0,
        0.20,
        0.30,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    values[21] = 0.02 + frame_index / 1000.0
    values[22:29] = np.linspace(-0.01, -0.07, 7) - frame_index / 1000.0
    values[36:43] = [
        0.40 + frame_index / 1000.0,
        -0.20,
        0.30,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    values[43] = 0.07 - frame_index / 1000.0
    return values


def _yam_state(frame_index: int) -> np.ndarray:
    values = np.zeros(20, dtype=np.float32)
    values[0:6] = np.linspace(0.001, 0.006, 6) + frame_index / 10000.0
    # Yam publishes xyz + xyzw; the recorder converts this to canonical wxyz.
    values[6:13] = [0.0, 0.0, 0.2 + frame_index / 1000.0, 0, 0, 0, 1]
    values[13:20] = [0.0, 0.0, 0.2 + frame_index / 1000.0, 0, 0, 0, 1]
    return values


def _make_saver() -> DataSaverConsumer:
    saver = DataSaverConsumer.__new__(DataSaverConsumer)
    saver.devices = {
        "right-camera": {
            "type": "WebCameraOpenPi",
            "name": "right-camera",
            "shape": [48, 64, 3],
        },
        "left-camera": {
            "type": "WebCameraOpenPi",
            "name": "left-camera",
            "shape": [48, 64, 3],
        },
        "head-camera": {
            "type": "WebCameraOpenPi",
            "name": "head-camera",
            "shape": [48, 64, 3],
        },
        "RizonRobot_2": {"type": "RizonRobot", "shape": [44]},
        "YamHeadRobot_0": {"type": "YamHeadRobot", "shape": [20]},
        "RizonRobot_2_control": {"type": "ControlSHM", "shape": [16]},
    }
    saver.data_format = NEDF3_DATA_FORMAT
    saver.image_size = None
    saver.task_name = "nedf3 recorder contract probe"
    saver.robot_type = "Rizon"
    saver.camera_id_mapping = {}
    saver.camera_id_overrides = {
        "right-camera": NEDF3_CAMERA_IDS["right"],
        "left-camera": NEDF3_CAMERA_IDS["left"],
        "head-camera": NEDF3_CAMERA_IDS["head"],
    }
    saver.strict_camera_ids = True
    saver.nedf3_camera_topics = {}
    saver.camera_params_written = {}
    saver.topics_config = None
    saver.nedf_writer = None
    saver.nedf_writer_context = None
    saver.current_episode_dir = None
    saver.episode_active = False
    saver.device_data_counts = {}
    saver.write_queue = queue.Queue(maxsize=128)
    saver.write_thread_running = False
    saver.summary_shm = None
    saver.reference_shm_name = "RizonRobot_2_control"
    return saver


def test_real_nedf3_async_recorder_and_conversion_receipt(tmp_path: Path) -> None:
    configured_output = os.environ.get("NEDF3_RECORDER_PROBE_OUTPUT", "").strip()
    episode_dir = (
        Path(configured_output).expanduser().resolve()
        if configured_output
        else tmp_path / "episode_0000"
    )
    assert not episode_dir.exists(), f"probe output already exists: {episode_dir}"

    saver = _make_saver()
    committed: list[dict[str, object]] = []
    failures: list[tuple[dict[str, object], BaseException]] = []
    saver._on_write_committed = committed.append
    saver._on_write_failed = lambda task, error: failures.append((task, error))
    saver.open_episode_streams(str(episode_dir))

    saver.write_thread_running = True
    worker = threading.Thread(target=saver._async_write_worker, daemon=True)
    worker.start()
    counts_before_close: dict[str, int] = {}
    try:
        for frame_index in range(FRAME_COUNT):
            timestamp_ns = 1_000_000_000 + frame_index * 33_333_333
            for camera_index, camera_key in enumerate(
                ("right-camera", "left-camera", "head-camera")
            ):
                image = np.full(
                    (48, 64, 3),
                    20 + frame_index * 10 + camera_index * 40,
                    dtype=np.uint8,
                )
                saver.write_queue.put(
                    _task(camera_key, "WebCameraOpenPi", timestamp_ns, image)
                )
            saver.write_queue.put(
                _task("RizonRobot_2", "RizonRobot", timestamp_ns, _rizon_state(frame_index))
            )
            saver.write_queue.put(
                _task("YamHeadRobot_0", "YamHeadRobot", timestamp_ns, _yam_state(frame_index))
            )
            saver.write_queue.put(
                _task(
                    "RizonRobot_2_control",
                    "ControlSHM",
                    timestamp_ns,
                    np.asarray(
                        [
                            0.4, 0.2, 0.3, 0, 0, 0, 1, 0.02,
                            0.4, -0.2, 0.3, 0, 0, 0, 1, 0.07,
                        ],
                        dtype=np.float32,
                    ),
                )
            )

        deadline = time.monotonic() + 10.0
        while saver.write_queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        assert saver.write_queue.unfinished_tasks == 0
        counts_before_close = dict(saver.device_data_counts)
    finally:
        saver.write_thread_running = False
        worker.join(timeout=5.0)
        saver.close_episode_streams()

    assert not worker.is_alive()
    assert failures == []
    assert set(counts_before_close.values()) == {FRAME_COUNT}
    committed_types = Counter(str(task["device_type"]) for task in committed)
    assert committed_types == {
        "WebCameraOpenPi": FRAME_COUNT * 3,
        "RizonRobot": FRAME_COUNT,
        "YamHeadRobot": FRAME_COUNT,
        "ControlSHM": FRAME_COUNT,
    }

    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    assert tuple(metadata["record_topic_list"]) == NEDF3_RECORD_TOPIC_LIST
    assert metadata["main_camera"] == NEDF3_CAMERA_IDS["right"]
    reader = NEDFFactory.get_reader(
        NEDFReaderConfig(metadata_file_path=str(episode_dir / "metadata.json"))
    )
    topic_timestamps = reader.get_raw_topics_timestamps()
    required_data_topics = {
        *(f"nmx/hal/camera/{camera_id}/rgbd" for camera_id in NEDF3_CAMERA_IDS.values()),
        *(NEDF3_ARM_TOPICS[side][kind] for side in ("left", "right") for kind in ("tcp/cmd", "tcp/state", "joint/state", "joint/vel", "eef_wrench", "extra")),
        *(NEDF3_HAND_TOPICS[side][kind] for side in ("left", "right") for kind in ("cmd", "state", "extra")),
        *(NEDF3_HEAD_TOPICS[kind] for kind in ("cmd", "state", "joint/state", "extra")),
    }
    assert set(topic_timestamps) == required_data_topics
    assert all(
        len(topic_timestamps[topic]) == FRAME_COUNT
        for topic in required_data_topics
    )
    left_command = next(
        reader.iter_decoded_messages(NEDF3_ARM_TOPICS["left"]["tcp/cmd"])
    )
    np.testing.assert_allclose(
        extract_pose_wxyz(left_command), [0.4, 0.2, 0.3, 1, 0, 0, 0]
    )
    left_hand_command = next(
        reader.iter_decoded_messages(NEDF3_HAND_TOPICS["left"]["cmd"])
    )
    assert extract_gripper_width(left_hand_command) == pytest.approx(0.02)
    left_joint_state = next(
        reader.iter_decoded_messages(NEDF3_ARM_TOPICS["left"]["joint/state"])
    )
    np.testing.assert_allclose(
        extract_joint_state_positions(left_joint_state),
        np.linspace(0.01, 0.07, 7),
    )
    right_joint_state = next(
        reader.iter_decoded_messages(NEDF3_ARM_TOPICS["right"]["joint/state"])
    )
    np.testing.assert_allclose(
        extract_joint_state_positions(right_joint_state),
        np.linspace(-0.01, -0.07, 7),
    )

    episode_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(episode_dir)))
    (episode_dir / "episode_result.json").write_text(
        json.dumps(
            {
                "episode_result": "success",
                "is_success": True,
                "decision": "accept",
                "episode_id": episode_id,
                "episode_index": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (episode_dir / "dense_labels.jsonl").write_text(
        json.dumps(
            {
                "label": "accept",
                "start_frame": 0,
                "end_frame": FRAME_COUNT,
                "interval": "[start_frame,end_frame)",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = write_conversion_source_receipt(
        episode_dir,
        episode_id=episode_id,
        episode_index=0,
        result="success",
        config_name="nedf3-recorder-contract-probe",
    )
    assert receipt["data_format"] == NEDF3_DATA_FORMAT
    assert receipt["fingerprint"]
    assert json.loads(
        (episode_dir / ".conversion_source.json").read_text(encoding="utf-8")
    ) == receipt
    print(
        json.dumps(
            {
                "episode": str(episode_dir),
                "fingerprint": receipt["fingerprint"],
                "data_topics": len(topic_timestamps),
                "messages_per_topic": FRAME_COUNT,
                "control_commits": committed_types["ControlSHM"],
            },
            sort_keys=True,
        )
    )
