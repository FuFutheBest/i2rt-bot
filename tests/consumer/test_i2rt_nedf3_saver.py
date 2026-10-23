"""Hardware-free tests for the terminal I2RT NEDF3 recorder."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from consumer.io.async_data_saver_i2rt_NEDF3 import (
    DeviceFrame,
    EpisodeSignalWatcher,
    I2RTNEDF3Saver,
    SaverConfig,
    apply_runtime_overrides,
    arm_state_message,
    hand_state_message,
    prepare_camera_image,
    rgbd_message,
    run_terminal,
    select_main_camera,
    split_i2rt_state,
)
from consumer.io.nedf3_i2rt_contract import (
    NEDF3_CAMERA_TOPICS,
    NEDF3_I2RT_HAND_TOPICS,
    NEDF3_I2RT_JOINT_TOPICS,
    NEDF3_RECORD_TOPICS,
    validate_topics,
)
from devices.base import BaseDevice


class MemoryWriter:
    def __init__(self) -> None:
        self.records: list[tuple[str, Any, int]] = []
        self.closed = False

    def write_message(self, topic: str, message: Any, timestamp_ns: int) -> None:
        self.records.append((topic, message, timestamp_ns))

    def close(self) -> None:
        self.closed = True


def _publisher(name: str, shape: tuple[int, ...], dtype: str) -> BaseDevice:
    dimension = int(np.prod(shape))
    device = BaseDevice(
        device_class_id=0,
        device_name=name,
        state_feedback={
            "fps": 30.0,
            "buffer_size": 4,
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


def _config(
    tmp_path: Path, names: dict[str, str], *, synthetic_head: bool = False
) -> SaverConfig:
    return SaverConfig(
        output_dir=tmp_path,
        sample_hz=25.0,
        queue_size=32,
        jpeg_quality=85,
        leader_shm=f"{names['leader']}_0_data",
        follower_shm=f"{names['follower']}_0_data",
        camera_shm={
            "right": f"{names['right']}_0_data",
            "left": f"{names['left']}_0_data",
            "head": None if synthetic_head else f"{names['head']}_0_data",
        },
        task="unreal_test",
        task_description="simulated I2RT streams",
        synthetic_camera_shapes={"head": (480, 640, 3)} if synthetic_head else {},
    )


def test_i2rt_joint_and_hand_messages_preserve_layout() -> None:
    values = np.arange(14, dtype=np.float64) + 0.25
    state = split_i2rt_state(values)

    np.testing.assert_array_equal(state.left_joints, values[:6])
    assert state.left_gripper == values[6]
    np.testing.assert_array_equal(state.right_joints, values[7:13])
    assert state.right_gripper == values[13]

    arm = arm_state_message(state.right_joints)
    assert [joint.name for joint in arm.joints] == [f"joint{i}" for i in range(6)]
    np.testing.assert_allclose([joint.pos for joint in arm.joints], values[7:13])
    hand = hand_state_message(state.right_gripper)
    assert [(joint.name, joint.pos) for joint in hand.joints] == [
        ("gripper", values[13])
    ]


@pytest.mark.parametrize(
    "values",
    ([0.0] * 13, [0.0] * 15, [0.0] * 13 + [float("nan")]),
)
def test_i2rt_state_rejects_invalid_vectors(values: list[float]) -> None:
    with pytest.raises(ValueError):
        split_i2rt_state(values)


@pytest.mark.parametrize("role", ("left", "right"))
@pytest.mark.parametrize("shape", ((360, 640, 3), (480, 640, 3)))
def test_wrist_camera_keeps_captured_frame(
    role: str, shape: tuple[int, int, int]
) -> None:
    image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)

    prepared = prepare_camera_image(role, image)

    assert prepared.shape == shape
    assert prepared.dtype == np.uint8
    np.testing.assert_array_equal(prepared, image)


def test_head_camera_keeps_640_by_480_frame() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    prepared = prepare_camera_image("head", image)

    assert prepared.shape == (480, 640, 3)
    with pytest.raises(ValueError, match="must have one of"):
        prepare_camera_image("head", np.zeros((360, 640, 3), dtype=np.uint8))


def test_rgbd_jpeg_preserves_rgb_channel_order() -> None:
    import cv2

    source_rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    source_rgb[:, :] = (240, 30, 10)

    message = rgbd_message(source_rgb, jpeg_quality=100)
    decoded_bgr = cv2.imdecode(
        np.frombuffer(message.rgb.data, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    assert decoded_bgr is not None
    mean_bgr = decoded_bgr.reshape(-1, 3).mean(axis=0)
    assert mean_bgr[2] > 220
    assert mean_bgr[0] < 30


def test_i2rt_contract_requires_exact_unique_topics() -> None:
    assert validate_topics(NEDF3_RECORD_TOPICS) == NEDF3_RECORD_TOPICS
    with pytest.raises(ValueError, match="duplicates"):
        validate_topics((*NEDF3_RECORD_TOPICS, NEDF3_RECORD_TOPICS[-1]))
    with pytest.raises(ValueError, match="differs"):
        validate_topics(NEDF3_RECORD_TOPICS[:-1])
    with pytest.raises(ValueError, match="unexpected"):
        validate_topics((*NEDF3_RECORD_TOPICS, "unexpected"))


def test_legacy_sample_hz_applies_to_all_streams(tmp_path: Path) -> None:
    config = SaverConfig(
        output_dir=tmp_path,
        sample_hz=25.0,
        queue_size=4,
        jpeg_quality=90,
        leader_shm="leader",
        follower_shm="follower",
        camera_shm={role: role for role in NEDF3_CAMERA_TOPICS},
        task="legacy",
        task_description="",
    )

    assert config.resolved_gripper_sample_hz == 25.0
    assert config.resolved_camera_sample_hz == 25.0
    assert config.polling_hz == 50.0
    assert config.camera_polling_hz == 50.0


def test_dm3_runtime_overrides_are_written_to_episode_metadata(
    tmp_path: Path,
) -> None:
    writer = MemoryWriter()
    captured: dict[str, Any] = {}
    base = SaverConfig(
        output_dir=tmp_path / "default",
        sample_hz=25.0,
        queue_size=4,
        jpeg_quality=90,
        leader_shm="unused_leader",
        follower_shm="unused_follower",
        camera_shm={role: f"unused_{role}" for role in ("right", "left", "head")},
        task="default_task",
        task_description="default description",
    )
    config = apply_runtime_overrides(
        base,
        output_dir=str(tmp_path / "dm3_task"),
        task_name="DM3 task",
        task_description="pick and place",
        dm3_user_id="user-17",
        dm3_user_name="Operator",
        task_execution_id="execution-23",
    )

    def writer_factory(_path: Path, metadata: Any) -> MemoryWriter:
        captured["metadata"] = metadata
        return writer

    saver = I2RTNEDF3Saver(config, writer_factory=writer_factory)
    saver.start_episode()
    saver.stop_episode(discard=True)

    metadata = captured["metadata"]
    assert metadata.task == "DM3 task"
    assert metadata.task_description == "pick and place"
    assert metadata.extra["user_id"] == "user-17"
    assert metadata.extra["user_name"] == "Operator"
    assert metadata.extra["task_execution_id"] == "execution-23"
    assert metadata.extra["camera_color_space"] == {
        "shared_memory": "rgb",
        "jpeg": "rgb",
        "opencv_encoder_input": "bgr",
    }
    assert metadata.extra["camera_processing"] == {
        "head": {"transform": "none"},
        "wrist": {"transform": "none"},
    }
    assert metadata.main_camera == "camera_2"


def test_synthetic_head_selects_real_right_wrist_as_main_camera() -> None:
    assert select_main_camera({"head"}) == "camera_0"
    assert select_main_camera(set()) == "camera_2"


def test_simulated_shm_episode_maps_all_canonical_topics(tmp_path: Path) -> None:
    suffix = uuid.uuid4().hex[:10]
    names = {
        key: f"I2RTTest_{key}_{suffix}"
        for key in ("leader", "follower", "right", "left")
    }
    publishers = {
        "leader": _publisher(names["leader"], (14,), "float64"),
        "follower": _publisher(names["follower"], (14,), "float64"),
        "right": _publisher(names["right"], (480, 640, 3), "uint8"),
        "left": _publisher(names["left"], (480, 640, 3), "uint8"),
    }
    writer = MemoryWriter()
    saver = I2RTNEDF3Saver(
        _config(tmp_path, names, synthetic_head=True),
        writer_factory=lambda _path, _metadata: writer,
    )
    try:
        saver.connect()
        saver.start_episode()
        timestamp_ns = time.time_ns()
        leader = np.arange(14, dtype=np.float64)
        follower = leader + 100.0
        publishers["leader"]._write_array_to_shm_with_timestamp(leader, timestamp_ns)
        publishers["follower"]._write_array_to_shm_with_timestamp(
            follower, timestamp_ns + 1
        )
        for index, role in enumerate(("right", "left")):
            image = np.full((480, 640, 3), 20 + index, dtype=np.uint8)
            publishers[role]._write_array_to_shm_with_timestamp(
                image, timestamp_ns + 2 + index
            )

        assert saver.collect_once() == 5
        destination = saver.stop_episode()
        assert destination == tmp_path / "episode_0000"
        assert destination.is_dir()
        assert writer.closed

        topics = [record[0] for record in writer.records]
        assert set(topics) == set(NEDF3_RECORD_TOPICS)
        assert len(topics) == len(NEDF3_RECORD_TOPICS)
        by_topic = {topic: message for topic, message, _timestamp in writer.records}
        leader_right = by_topic[NEDF3_I2RT_JOINT_TOPICS["leader"]["right"]]
        follower_left = by_topic[NEDF3_I2RT_JOINT_TOPICS["follower"]["left"]]
        np.testing.assert_allclose(
            [joint.pos for joint in leader_right.joints], leader[7:13]
        )
        np.testing.assert_allclose(
            [joint.pos for joint in follower_left.joints], follower[:6]
        )
        leader_hand = by_topic[NEDF3_I2RT_HAND_TOPICS["leader"]["left"]]
        assert leader_hand.joints[0].pos == leader[6]
        from consumer.io.nedf3_i2rt_contract import NEDF3_CAMERA_TOPICS

        import cv2

        head_message = by_topic[NEDF3_CAMERA_TOPICS["head"]]
        decoded_head = cv2.imdecode(
            np.frombuffer(head_message.rgb.data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded_head.shape == (480, 640, 3)
        assert np.count_nonzero(decoded_head) == 0
        for role in ("right", "left"):
            wrist_message = by_topic[NEDF3_CAMERA_TOPICS[role]]
            decoded_wrist = cv2.imdecode(
                np.frombuffer(wrist_message.rgb.data, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            assert decoded_wrist.shape == (480, 640, 3)

        stats = json.loads((destination / "episode_stats.json").read_text())
        assert stats["dropped_batches"] == 0
        assert all(count == 1 for count in stats["topic_counts"].values())
    finally:
        saver.close()
        for publisher in publishers.values():
            publisher._cleanup_shared_memory()


def test_real_head_camera_frame_is_recorded(tmp_path: Path) -> None:
    suffix = uuid.uuid4().hex[:10]
    names = {
        key: f"I2RTTest_{key}_{suffix}"
        for key in ("leader", "follower", "right", "left", "head")
    }
    publishers = {
        "leader": _publisher(names["leader"], (14,), "float64"),
        "follower": _publisher(names["follower"], (14,), "float64"),
        "right": _publisher(names["right"], (360, 640, 3), "uint8"),
        "left": _publisher(names["left"], (360, 640, 3), "uint8"),
        "head": _publisher(names["head"], (480, 640, 3), "uint8"),
    }
    writer = MemoryWriter()
    saver = I2RTNEDF3Saver(
        _config(tmp_path, names),
        writer_factory=lambda _path, _metadata: writer,
    )
    try:
        saver.connect()
        saver.start_episode()
        timestamp_ns = time.time_ns()
        publishers["leader"]._write_array_to_shm_with_timestamp(
            np.arange(14, dtype=np.float64), timestamp_ns
        )
        publishers["follower"]._write_array_to_shm_with_timestamp(
            np.arange(14, dtype=np.float64), timestamp_ns + 1
        )
        for index, role in enumerate(("right", "left")):
            publishers[role]._write_array_to_shm_with_timestamp(
                np.full((360, 640, 3), 30 + index, dtype=np.uint8),
                timestamp_ns + 2 + index,
            )
        publishers["head"]._write_array_to_shm_with_timestamp(
            np.full((480, 640, 3), 120, dtype=np.uint8), timestamp_ns + 4
        )

        assert saver.collect_once() == 5
        saver.stop_episode()
        head_topic = NEDF3_CAMERA_TOPICS["head"]
        head_message = next(
            message for topic, message, _ in writer.records if topic == head_topic
        )
        import cv2

        decoded = cv2.imdecode(
            np.frombuffer(head_message.rgb.data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded.shape == (480, 640, 3)
        assert np.allclose(decoded, 120, atol=2)
    finally:
        saver.close()
        for publisher in publishers.values():
            publisher._cleanup_shared_memory()


def test_recorder_samples_arm_at_90_hz_and_gripper_and_cameras_at_30_hz(
    tmp_path: Path,
) -> None:
    class IncrementingReader:
        def __init__(self, data: np.ndarray) -> None:
            self.data = data
            self.timestamp_ns = 0

        def read_latest(self) -> DeviceFrame:
            self.timestamp_ns += 1
            return DeviceFrame(self.timestamp_ns, self.data.copy())

    class Camera30HzReader:
        def __init__(self, data: np.ndarray) -> None:
            self.data = data
            self.read_count = 0

        def read_latest(self) -> DeviceFrame:
            self.read_count += 1
            timestamp_ns = (self.read_count - 1) // 3 + 1
            return DeviceFrame(timestamp_ns, self.data.copy())

    writer = MemoryWriter()
    config = SaverConfig(
        output_dir=tmp_path,
        sample_hz=90.0,
        gripper_sample_hz=30.0,
        camera_sample_hz=30.0,
        queue_size=512,
        jpeg_quality=85,
        leader_shm="leader",
        follower_shm="follower",
        camera_shm={"right": "right", "left": "left", "head": None},
        task="frequency_test",
        task_description="",
        synthetic_camera_shapes={"head": (480, 640, 3)},
    )
    saver = I2RTNEDF3Saver(
        config, writer_factory=lambda _path, _metadata: writer
    )
    robot_state = np.arange(14, dtype=np.float64)
    wrist_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    saver.readers = {
        "leader": IncrementingReader(robot_state),
        "follower": IncrementingReader(robot_state),
        "camera:right": Camera30HzReader(wrist_frame),
        "camera:left": Camera30HzReader(wrist_frame),
    }

    saver.start_episode()
    for sample_index in range(90):
        saver.collect_once(now_ns=sample_index * 1_000_000_000 // 90)
    destination = saver.stop_episode()

    stats = json.loads((destination / "episode_stats.json").read_text())
    for topics in NEDF3_I2RT_JOINT_TOPICS.values():
        assert all(stats["topic_counts"][topic] == 90 for topic in topics.values())
    for topics in NEDF3_I2RT_HAND_TOPICS.values():
        assert all(stats["topic_counts"][topic] == 30 for topic in topics.values())
    assert all(
        stats["topic_counts"][topic] == 30
        for topic in NEDF3_CAMERA_TOPICS.values()
    )
    assert saver.readers["camera:right"].read_count == 90
    assert saver.readers["camera:left"].read_count == 90


def test_recorder_retries_due_arm_sample_until_source_advances(tmp_path: Path) -> None:
    class SourceRateReader:
        def __init__(self, data: np.ndarray, clock_ns: list[int]) -> None:
            self.data = data
            self.clock_ns = clock_ns

        def read_latest(self) -> DeviceFrame:
            source_index = self.clock_ns[0] * 195 // 2_000_000_000
            return DeviceFrame(source_index + 1, self.data.copy())

    class StaticCameraReader:
        def __init__(self, data: np.ndarray) -> None:
            self.data = data

        def read_latest(self) -> DeviceFrame:
            return DeviceFrame(1, self.data.copy())

    clock_ns = [0]
    writer = MemoryWriter()
    config = SaverConfig(
        output_dir=tmp_path,
        sample_hz=90.0,
        gripper_sample_hz=30.0,
        camera_sample_hz=30.0,
        queue_size=512,
        jpeg_quality=85,
        leader_shm="leader",
        follower_shm="follower",
        camera_shm={"right": "right", "left": "left", "head": None},
        task="source_retry_test",
        task_description="",
        synthetic_camera_shapes={"head": (480, 640, 3)},
    )
    robot_state = np.arange(14, dtype=np.float64)
    wrist_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    saver = I2RTNEDF3Saver(config, writer_factory=lambda _path, _metadata: writer)
    saver.readers = {
        "leader": SourceRateReader(robot_state, clock_ns),
        "follower": SourceRateReader(robot_state, clock_ns),
        "camera:right": StaticCameraReader(wrist_frame),
        "camera:left": StaticCameraReader(wrist_frame),
    }

    saver.start_episode()
    polling_samples = round(config.polling_hz)
    for sample_index in range(polling_samples):
        clock_ns[0] = sample_index * 1_000_000_000 // polling_samples
        saver.collect_once(now_ns=clock_ns[0])
    destination = saver.stop_episode()

    stats = json.loads((destination / "episode_stats.json").read_text())
    for topics in NEDF3_I2RT_JOINT_TOPICS.values():
        assert all(stats["topic_counts"][topic] == 90 for topic in topics.values())


def test_terminal_scheduler_accounts_for_collection_work(tmp_path: Path) -> None:
    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

        def advance(self, seconds: float) -> None:
            self.now += seconds

    class FakeSaver:
        def __init__(self, clock: FakeClock) -> None:
            self.clock = clock
            self.config = SaverConfig(
                output_dir=tmp_path,
                sample_hz=90.0,
                gripper_sample_hz=30.0,
                camera_sample_hz=30.0,
                queue_size=4,
                jpeg_quality=90,
                leader_shm="leader",
                follower_shm="follower",
                camera_shm={role: role for role in NEDF3_CAMERA_TOPICS},
                task="scheduler_test",
                task_description="",
                episode_signal_pattern=str(tmp_path / "episode_{signal}"),
            )
            self.active = False
            self.sample_times: list[float] = []
            self.closed = False

        def connect(self) -> None:
            pass

        def start_episode(self) -> Path:
            self.active = True
            return tmp_path / "episode_0000_inprogress"

        def collect_once(self) -> None:
            self.sample_times.append(self.clock())
            self.clock.advance(0.002)

        def stop_episode(self, *, discard: bool = False) -> Path:
            self.active = False
            return tmp_path / ("discarded" if discard else "episode_0000")

        def close(self) -> None:
            self.closed = True

    clock = FakeClock()
    saver = FakeSaver(clock)

    def wait_for_input(timeout: float) -> bool:
        clock.advance(timeout)
        return False

    assert run_terminal(
        saver,
        auto_start=True,
        duration=1.0,
        clock=clock,
        wait_for_input=wait_for_input,
    ) == 0

    intervals = np.diff(saver.sample_times)
    assert len(saver.sample_times) in {180, 181}
    np.testing.assert_allclose(intervals, 1.0 / 180.0, atol=1e-12)
    assert saver.closed


def test_writer_failure_quarantines_episode(tmp_path: Path) -> None:
    class FailingWriter(MemoryWriter):
        def write_message(self, topic: str, message: Any, timestamp_ns: int) -> None:
            raise OSError("simulated disk failure")

    writer = FailingWriter()
    config = SaverConfig(
        output_dir=tmp_path,
        sample_hz=25.0,
        queue_size=4,
        jpeg_quality=90,
        leader_shm="unused_leader",
        follower_shm="unused_follower",
        camera_shm={role: f"unused_{role}" for role in ("right", "left", "head")},
        task="failure",
        task_description="",
    )
    saver = I2RTNEDF3Saver(config, writer_factory=lambda _path, _metadata: writer)
    saver.start_episode()
    destination = saver.stop_episode()

    assert destination.parent == tmp_path / "quarantine"
    reason = json.loads((destination / "quarantine_reason.json").read_text())
    assert "simulated disk failure" in reason["writer_error"]


def test_start_recovers_interrupted_episode_and_advances_index(tmp_path: Path) -> None:
    interrupted = tmp_path / "episode_0004_inprogress"
    interrupted.mkdir()
    (interrupted / "partial").write_text("incomplete", encoding="utf-8")
    writer = MemoryWriter()
    config = SaverConfig(
        output_dir=tmp_path,
        sample_hz=25.0,
        queue_size=4,
        jpeg_quality=90,
        leader_shm="unused_leader",
        follower_shm="unused_follower",
        camera_shm={role: f"unused_{role}" for role in ("right", "left", "head")},
        task="recovery",
        task_description="",
    )
    saver = I2RTNEDF3Saver(config, writer_factory=lambda _path, _metadata: writer)

    started = saver.start_episode()
    saver.stop_episode(discard=True)

    assert started.name == "episode_0005_inprogress"
    recovered = tmp_path / "quarantine" / "episode_0004_interrupted"
    assert (recovered / "partial").read_text(encoding="utf-8") == "incomplete"
    reason = json.loads((recovered / "quarantine_reason.json").read_text())
    assert reason["reason"] == "recorder exited before episode finalization"


def test_episode_signal_watcher_orders_and_consumes_commands(tmp_path: Path) -> None:
    pattern = str(tmp_path / "episode_{signal}")
    watcher = EpisodeSignalWatcher(pattern)
    Path(pattern.format(signal="discard")).write_text("20\n", encoding="ascii")
    Path(pattern.format(signal="toggle")).write_text("10\n", encoding="ascii")

    assert watcher.drain() == ["toggle", "discard"]
    assert watcher.drain() == []
