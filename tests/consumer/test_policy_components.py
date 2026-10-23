"""Unit tests for PolicyConnector split components.

Author: OpenAI Assistant
"""

from __future__ import annotations

import json
import queue
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

from consumer.inference.common.episode_controller import EpisodeController
from consumer.inference.common.observation_builder import ObservationBuilder
from consumer.inference.common.policy_connector import (
    PolicyConnector,
    _resolve_rollout_python_path,
)
from consumer.inference.common.policy_loop import PolicyLoop
from consumer.inference.common.policy_output_writer import PolicyOutputWriter
from consumer.inference.common.policy_runtime import PolicyRuntime
from consumer.inference.common.policy_runtime_io import PolicyRuntimeIO
from consumer.inference.common.policy_visualization import PolicyVisualization
from consumer.inference.common.rollout_controller import RolloutController
from consumer.inference.common.value_prediction import ValuePredictionWorker
from consumer.policy_history import PolicyHistoryManager
from utils.policy_control_config import ControlTargetConfig
from utils.shm_utils import (
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    MANAGER_HEADER_SIZE,
    POLICY_HEADER_SIZE,
    SUMMARY_FRAME_HEADER_SIZE,
    calculate_policy_shm_size,
    calculate_summary_shm_size,
    get_policy_data_offset,
    get_summary_frame_start_offset,
    pack_manager_header,
    pack_summary_frame_header,
    unpack_frame_header,
    unpack_policy_header,
)


class DummyPolicy:
    """Minimal policy used to test PolicyRuntime without model dependencies."""

    def __init__(self, scale: float = 1.0, **_: Any) -> None:
        self.scale = scale
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def predict(
        self,
        observation: Any,
        action_configs: List[Dict[str, Any]],
        chunk_length: int = 1,
        **kwargs: Any,
    ) -> Dict[str, np.ndarray]:
        return {
            cfg["device_name"]: np.full((chunk_length, cfg["action_dim"]), self.scale, dtype=np.float64)
            for cfg in action_configs
        }


class LegacyPredictPolicy:
    """Policy that keeps the pre-RTC three-argument prediction contract."""

    def load(self) -> None:
        pass

    def predict(
        self,
        observation: Any,
        action_configs: List[Dict[str, Any]],
        chunk_length: int = 1,
    ) -> Dict[str, np.ndarray]:
        del observation
        return {
            cfg["device_name"]: np.zeros(
                (chunk_length, cfg["action_dim"]), dtype=np.float64
            )
            for cfg in action_configs
        }


def test_control_panel_recording_python_overrides_native_config() -> None:
    config = {"python_path": "/native/recording/python"}

    assert _resolve_rollout_python_path(
        config,
        {
            "SUPERINFERENCE_RECORDER_PYTHON": "/control-panel/recording/python",
            "SUPERINFERENCE_RUNTIME_REGISTRY": "/missing/runtime_profiles.json",
        },
    ) == "/control-panel/recording/python"


def test_native_rollout_keeps_configured_recording_python() -> None:
    assert _resolve_rollout_python_path(
        {"python_path": "/native/recording/python"},
        {"SUPERINFERENCE_RUNTIME_REGISTRY": "/missing/runtime_profiles.json"},
    ) == "/native/recording/python"


def test_native_rollout_defaults_to_current_python() -> None:
    assert _resolve_rollout_python_path(
        {},
        {"SUPERINFERENCE_RUNTIME_REGISTRY": "/missing/runtime_profiles.json"},
    ) == sys.executable


class FakePolicyOutputShm:
    """Small in-memory stand-in for PolicyOutputSHM."""

    def __init__(self) -> None:
        self.enabled = True
        self.created = False
        self.shape = None
        self.writes: list[tuple[np.ndarray, int]] = []
        self.multi_writes: list[tuple[list[np.ndarray], np.ndarray]] = []

    def create(self, shape: tuple[int, ...]) -> None:
        self.created = True
        self.shape = tuple(shape)

    def write(self, value: np.ndarray, timestamp_ns: int) -> None:
        self.writes.append((value.copy(), int(timestamp_ns)))

    def write_multi(self, values: list[np.ndarray], timestamps: np.ndarray) -> None:
        self.multi_writes.append(([value.copy() for value in values], timestamps.copy()))

    def get_config(self) -> dict[str, object]:
        return {"shm": "FakeOutput", "shape": list(self.shape or (1,)), "dtype": "float64"}


class FakeActionStateLogger:
    """Capture visualization log calls for unit tests."""

    def __init__(self) -> None:
        self.states: list[dict[str, Any]] = []
        self.metas: list[dict[str, Any]] = []

    def log_state(self, **kwargs: Any) -> None:
        self.states.append(kwargs)

    def log_meta(self, event_type: str, **kwargs: Any) -> None:
        self.metas.append({"event_type": event_type, **kwargs})


def _runtime_signal_config() -> dict[str, dict[str, str]]:
    """Minimal runtime signal config required by PolicyConnector.__init__."""
    required_signals = (
        "start_recording",
        "stop_recording",
        "reset_signal",
        "robot_home_index",
        "episode_result",
        "policy_runner_reset_command",
        "teleoperator_reset_command",
        "rollout_ready",
        "robot_reconnect",
        "robot_halt_command_pattern",
        "robot_halt_reply_pattern",
    )
    return {"signal_files": {key: f"/tmp/{key}" for key in required_signals}}


def _runtime_control_target(
    device: str,
    device_class_id: int,
    *,
    shape: tuple[int, ...] = (1,),
) -> dict[str, Any]:
    return ControlTargetConfig.from_mapping(
        {
            "endpoint": {
                "device": device,
                "device_class_id": device_class_id,
                "port": "command",
            },
            "shape": list(shape),
            "dtype": "float64",
        }
    ).runtime_mapping()


@pytest.mark.unit
def test_policy_connector_wires_common_components(monkeypatch: pytest.MonkeyPatch) -> None:
    """PolicyConnector should instantiate and expose the extracted common components."""
    import consumer.inference.common.policy_runtime as policy_runtime_module
    from consumer.inference.common.policy_connector import PolicyConnector

    monkeypatch.setitem(policy_runtime_module.POLICY_CLASSES, "DummyPolicy", DummyPolicy)

    connector = PolicyConnector(
        summary_shm_name="fake_summary",
        policy_class="DummyPolicy",
        policy_params={"scale": 1.5, "robot_device": "Robot_0"},
        obs_devices=["Camera_0", "Robot_0"],
        controls=[
            {
                "endpoint": {
                    "device": "Robot",
                    "device_class_id": 0,
                    "port": "command",
                },
                "shape": [3],
                "dtype": "float64",
            }
        ],
        fps=10.0,
        control_fps=20.0,
        chunk_length=4,
        master_device="Robot_0",
        runtime_config=_runtime_signal_config(),
    )

    assert isinstance(connector.policy_runtime, PolicyRuntime)
    assert isinstance(connector.observation_builder, ObservationBuilder)
    assert isinstance(connector.policy_output_writer, PolicyOutputWriter)
    assert isinstance(connector.policy_loop, PolicyLoop)
    assert isinstance(connector.runtime_io, PolicyRuntimeIO)
    assert isinstance(connector.episode_controller, EpisodeController)
    assert isinstance(connector.value_prediction_worker, ValuePredictionWorker)
    assert isinstance(connector.rollout_controller, RolloutController)
    assert isinstance(connector.policy_visualization, PolicyVisualization)
    assert isinstance(connector.policy_history_manager, PolicyHistoryManager)
    assert connector.policy is connector.policy_runtime.policy
    assert connector.observation_builder.master_device == "Robot_0"
    assert connector.observation_builder.read_all_device_data == connector.read_all_device_data
    assert connector.policy_output_writer.timestamp_logger is connector.timestamp_logger
    assert connector.action_configs == [
        {"device_name": "Robot_0", "action_dim": 3, "shm_name": "Robot_0_control"}
    ]


@pytest.mark.unit
def test_policy_connector_prepare_next_policy_input_delegates_history_update() -> None:
    """The shallow run-loop helper should assemble observations and call policy-history orchestration."""
    from consumer.inference.common.policy_connector import PolicyConnector

    obs_dict = {"Robot_0": np.array([1.0, 2.0])}
    history_calls: list[dict[str, Any]] = []

    connector = PolicyConnector.__new__(PolicyConnector)
    connector.summary_shm = None
    connector.inference_enabled = True
    connector.history_with_value = True
    connector.policy = SimpleNamespace(block=False)
    connector.policy_history_manager = SimpleNamespace(
        maybe_update_policy_history=lambda **kwargs: history_calls.append(kwargs) or True
    )
    connector.policy_loop = PolicyLoop(connector)
    connector._assemble_observation_dict = lambda: (obs_dict, 123)
    connector._prepare_policy_input = lambda latest_obs, latest_timestamp, history_with_value: {
        "latest_obs": latest_obs,
        "latest_timestamp": latest_timestamp,
        "history_with_value": history_with_value,
    }

    policy_input, obs_time, should_skip = PolicyConnector._prepare_next_policy_input(connector)

    assert should_skip is False
    assert obs_time == 123
    assert policy_input == {
        "latest_obs": obs_dict,
        "latest_timestamp": 123,
        "history_with_value": True,
    }
    assert history_calls == [
        {"latest_obs_time": 123, "block": False, "current_obs": obs_dict}
    ]


@pytest.mark.unit
def test_policy_connector_predict_action_chunk_delegates_runtime_with_context() -> None:
    """The prediction helper should pass RTC context into PolicyRuntime and log markers."""
    from consumer.inference.common.policy_connector import PolicyConnector

    contexts: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []
    runtime_calls: list[dict[str, Any]] = []
    expected_chunk = {"Robot_0": np.ones((2, 3), dtype=np.float64)}

    def predict(policy_input: dict[str, Any], **kwargs: Any) -> dict[str, np.ndarray]:
        runtime_calls.append({"policy_input": policy_input, **kwargs})
        return expected_chunk

    connector = PolicyConnector.__new__(PolicyConnector)
    connector.summary_shm = None
    connector.policy = SimpleNamespace(
        enable_rtc=True,
        set_prediction_context=lambda **kwargs: contexts.append(kwargs),
        last_prediction_timing_ns={
            "input_preprocess_ns": 1,
            "h2d_ns": 2,
            "cuda_forward_ns": 3,
            "output_sync_d2h_ns": 4,
            "postprocess_ns": 5,
            "policy_predict_ns": 6,
        },
    )
    connector.policy_runtime = SimpleNamespace(predict=predict)
    connector.action_configs = [{"device_name": "Robot_0", "action_dim": 3}]
    connector.chunk_length = 2
    connector._read_policy_action_index = lambda: 7
    connector._log_inference_visualizer_marker = lambda **kwargs: markers.append(kwargs)
    connector._maybe_log_policy_relative_xyz = lambda: None
    connector._maybe_log_policy_reference_xyz = lambda: None
    connector.policy_loop = PolicyLoop(connector)
    connector.runtime_io = PolicyRuntimeIO(connector)

    action_chunk, inference_start_ns = PolicyConnector._predict_action_chunk(
        connector,
        {"obs": "prepared"},
        obs_time=456,
        iteration_count=3,
    )

    assert action_chunk is expected_chunk
    assert isinstance(inference_start_ns, int)
    assert contexts == [{"timestamp_ns": 456}]
    assert runtime_calls == [
        {
            "policy_input": {"obs": "prepared"},
            "action_configs": [{"device_name": "Robot_0", "action_dim": 3}],
            "chunk_length": 2,
            "action_index": 7,
        }
    ]
    assert [marker["inference_event_type"] for marker in markers] == [
        "inference_start",
        "inference_end",
    ]
    assert markers[-1]["success"] is True
    assert {
        key: markers[-1][key]
        for key in (
            "policy_input_preprocess_ns",
            "policy_h2d_ns",
            "policy_cuda_forward_ns",
            "policy_output_sync_d2h_ns",
            "policy_postprocess_ns",
            "policy_policy_predict_ns",
        )
    } == {
        "policy_input_preprocess_ns": 1,
        "policy_h2d_ns": 2,
        "policy_cuda_forward_ns": 3,
        "policy_output_sync_d2h_ns": 4,
        "policy_postprocess_ns": 5,
        "policy_policy_predict_ns": 6,
    }


@pytest.mark.unit
def test_policy_inference_metrics_include_frequency_and_chunk_horizon() -> None:
    owner = SimpleNamespace(
        _inference_count=0,
        control_fps=15.0,
        chunk_length=24,
        fps=2.0,
    )
    loop = PolicyLoop(owner)

    first = loop._record_inference_runtime_metrics(1_000_000_000, 2_700_000_000)
    second = loop._record_inference_runtime_metrics(3_000_000_000, 3_100_000_000)

    assert first["action_chunk_horizon_s"] == 1.6
    assert first["inference_exceeds_action_horizon"] is True
    assert second["actual_policy_inference_hz"] == 0.5
    assert second["inference_count"] == 2


@pytest.mark.unit
def test_policy_connector_write_predicted_action_chunk_updates_history_and_recording() -> None:
    """The write helper should trim chunks, write them, and update history/recording state."""
    from consumer.inference.common.policy_connector import PolicyConnector

    merged_writes: list[tuple[np.ndarray, int]] = []
    sent_chunks: list[tuple[dict[str, np.ndarray], int]] = []
    start_recording_calls: list[bool] = []
    history_events: list[dict[str, Any]] = []
    action_chunk = {"Robot_0": np.arange(12, dtype=np.float64).reshape(4, 3)}

    connector = PolicyConnector.__new__(PolicyConnector)
    connector.summary_shm = None
    connector.chunk_length = 2
    connector.policy = SimpleNamespace(
        get_history_update_spec=lambda **kwargs: {"inference_start": kwargs["inference_start_timestamp_ns"]}
    )
    connector._infer_rollout_started = True
    connector.action_chunk_shm = SimpleNamespace(
        created=True,
        write=lambda value, timestamp: merged_writes.append((value.copy(), timestamp)),
    )
    connector._merge_action_chunks = lambda chunks: np.concatenate(list(chunks.values()), axis=1)
    connector._signal_start_recording = lambda: start_recording_calls.append(True)
    connector._write_action_chunk_dict = lambda chunks, observation_timestamp: (
        sent_chunks.append(({key: value.copy() for key, value in chunks.items()}, observation_timestamp))
        or 999
    )
    connector.policy_history_manager = SimpleNamespace(
        attach_action_chunk_timing=lambda chunks, **kwargs: history_events.append(
            {"event": "attach", "chunks": {key: value.copy() for key, value in chunks.items()}, **kwargs}
        ),
    )
    connector.policy_loop = PolicyLoop(connector)

    first_action_written = PolicyConnector._write_predicted_action_chunk(
        connector,
        action_chunk,
        obs_time=321,
        inference_start_ns=111,
        iteration_count=2,
        first_action_written=False,
    )

    assert first_action_written is True
    assert connector._pending_policy_history_spec == {"inference_start": 111}
    assert connector._policy_history_arrival_wait_start_ns is None
    assert start_recording_calls == [True]
    np.testing.assert_array_equal(action_chunk["Robot_0"], np.arange(6, dtype=np.float64).reshape(2, 3))
    np.testing.assert_array_equal(merged_writes[0][0], action_chunk["Robot_0"])
    assert merged_writes[0][1] == 321
    assert sent_chunks[0][1] == 321
    assert history_events[0]["event"] == "attach"
    assert history_events[0]["first_action_sent_timestamp_ns"] == 999
    assert len(history_events) == 1


@pytest.mark.unit
def test_policy_runtime_loads_and_predicts_with_fake_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """PolicyRuntime should isolate policy loading behind a small contract."""
    import consumer.inference.common.policy_runtime as policy_runtime_module

    monkeypatch.setitem(policy_runtime_module.POLICY_CLASSES, "DummyPolicy", DummyPolicy)

    runtime = PolicyRuntime("DummyPolicy", {"scale": 2.5})
    output = runtime.predict(
        {"obs": np.array([1.0])},
        [{"device_name": "Robot_0", "action_dim": 3}],
        chunk_length=2,
    )

    assert runtime.policy.loaded is True
    np.testing.assert_array_equal(output["Robot_0"], np.full((2, 3), 2.5))


@pytest.mark.unit
def test_policy_runtime_filters_optional_kwargs_for_legacy_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy policies must survive the common loop's RTC extension kwargs."""
    import consumer.inference.common.policy_runtime as policy_runtime_module

    monkeypatch.setitem(
        policy_runtime_module.POLICY_CLASSES,
        "LegacyPredictPolicy",
        LegacyPredictPolicy,
    )

    runtime = PolicyRuntime("LegacyPredictPolicy")
    output = runtime.predict(
        {"obs": np.array([1.0])},
        [{"device_name": "Robot_0", "action_dim": 2}],
        chunk_length=3,
        action_index=4,
    )

    np.testing.assert_array_equal(output["Robot_0"], np.zeros((3, 2)))


@pytest.mark.unit
def test_observation_builder_uses_master_device_timestamp() -> None:
    """ObservationBuilder should preserve existing master-device timestamp behavior."""
    logged: list[dict[str, Any]] = []

    def read_all_device_data(**kwargs: Any) -> tuple[int, dict[str, tuple[int, np.ndarray]]]:
        assert kwargs == {"include_frame_timestamp": True}
        return (
            300,
            {
                "Camera_0": (100, np.array([1.0])),
                "Robot_0": (200, np.array([2.0])),
            },
        )

    builder = ObservationBuilder(
        read_all_device_data=read_all_device_data,
        master_device="Robot_0",
        robot_state_logger=lambda **kwargs: logged.append(kwargs),
    )

    obs, timestamp = builder.assemble()

    assert timestamp == 200
    assert set(obs or {}) == {"Camera_0", "Robot_0"}
    assert len(logged) == 2


@pytest.mark.unit
def test_policy_output_writer_preserves_shape_and_timestamp_contract() -> None:
    """PolicyOutputWriter should write frame timestamp and action bytes to policy SHM."""
    action_chunk = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    frame_size = FRAME_HEADER_SIZE + action_chunk.nbytes
    policy_devices = [
        {
            "device_name": "Robot_0",
            "shape": action_chunk.shape,
            "frame_size": frame_size,
        }
    ]
    shm_size = calculate_policy_shm_size(1, [frame_size])
    fake_shm = SimpleNamespace(buf=bytearray(shm_size))
    timestamp_events: list[dict[str, Any]] = []
    writer = PolicyOutputWriter(
        policy_shm_getter=lambda: fake_shm,
        policy_devices_getter=lambda: policy_devices,
        reconnect_policy_shm=lambda: None,
        timestamp_logger=SimpleNamespace(
            log_event=lambda event_name, **kwargs: timestamp_events.append(
                {"event_name": event_name, **kwargs}
            )
        ),
    )

    write_timestamp = writer.write_action_chunk_dict({"Robot_0": action_chunk}, observation_timestamp=123)

    assert write_timestamp is not None
    assert unpack_policy_header(fake_shm.buf[:POLICY_HEADER_SIZE])["device_count"] == 1

    data_offset = get_policy_data_offset(0, 1, [frame_size])
    assert data_offset == MANAGER_HEADER_SIZE + DEVICE_HEADER_SIZE
    frame_header = fake_shm.buf[data_offset:data_offset + FRAME_HEADER_SIZE]
    assert unpack_frame_header(frame_header)["timestamp_ns"] == 123

    data_start = data_offset + FRAME_HEADER_SIZE
    data_end = data_start + action_chunk.nbytes
    restored = np.frombuffer(fake_shm.buf[data_start:data_end], dtype=np.float64).reshape(action_chunk.shape)
    np.testing.assert_array_equal(restored, action_chunk)
    assert timestamp_events[0]["event_name"] == "action_chunk_written"


@pytest.mark.unit
def test_value_prediction_worker_writes_scalar_value() -> None:
    """Scalar V-values should be reshaped and written to value SHM."""
    owner = SimpleNamespace(
        value_shm=FakePolicyOutputShm(),
        q_value_shm=FakePolicyOutputShm(),
        q_value_timestamps_shm=FakePolicyOutputShm(),
        action_step_interval_ns=20,
    )
    worker = ValuePredictionWorker(owner)

    worker._write_value_result(value_data=np.array(0.25), obs_time=100)

    assert owner.value_shm.created is True
    assert owner.value_shm.shape == (1,)
    assert len(owner.value_shm.writes) == 1
    value, timestamp_ns = owner.value_shm.writes[0]
    np.testing.assert_array_equal(value, np.array([0.25]))
    assert timestamp_ns == 100


@pytest.mark.unit
def test_value_prediction_worker_writes_q_values_and_timestamps() -> None:
    """Multi-step Q-values should be paired with action-step timestamps."""
    owner = SimpleNamespace(
        value_shm=FakePolicyOutputShm(),
        q_value_shm=FakePolicyOutputShm(),
        q_value_timestamps_shm=FakePolicyOutputShm(),
        action_step_interval_ns=20,
    )
    worker = ValuePredictionWorker(owner)
    q_values = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)

    worker._write_value_result(value_data=q_values, obs_time=100)

    assert owner.q_value_shm.created is True
    assert owner.q_value_timestamps_shm.created is True
    values, timestamps = owner.q_value_shm.multi_writes[0]
    np.testing.assert_array_equal(values[0], q_values)
    np.testing.assert_array_equal(values[1], np.array([100, 120, 140], dtype=np.int64))
    np.testing.assert_array_equal(timestamps, np.array([100, 120, 140], dtype=np.int64))


@pytest.mark.unit
def test_rollout_controller_touches_recording_signal_files(tmp_path) -> None:
    """Rollout recording signals should create the configured signal files."""
    owner = SimpleNamespace(
        _infer_rollout_started=True,
        signal_files={
            "start_recording": str(tmp_path / "start"),
            "stop_recording": str(tmp_path / "stop"),
        },
    )
    controller = RolloutController(owner)

    controller.signal_start_recording()
    controller.signal_stop_recording()

    assert (tmp_path / "start").exists()
    assert (tmp_path / "stop").exists()


@pytest.mark.unit
def test_rollout_controller_builds_control_shm_list_with_policy_outputs() -> None:
    """Rollout control SHMs should include only enabled, created policy outputs."""
    owner = SimpleNamespace(
        value_shm=FakePolicyOutputShm(),
        action_chunk_shm=FakePolicyOutputShm(),
        q_value_shm=FakePolicyOutputShm(),
        q_value_timestamps_shm=FakePolicyOutputShm(),
    )
    owner.value_shm.shape = (1,)
    owner.action_chunk_shm.shape = (2, 3)
    owner.q_value_shm.shape = (2, 1)
    owner.q_value_timestamps_shm.shape = (2,)
    owner.value_shm.created = True
    owner.action_chunk_shm.created = True
    owner.q_value_shm.created = True
    owner.q_value_timestamps_shm.created = True
    owner.q_value_shm.enabled = False
    owner.q_value_timestamps_shm.enabled = False
    controller = RolloutController(owner)

    control_shms = controller._build_control_shms_list(
        {
            "control_shms": [
                {
                    "endpoint": {"device": "Robot", "device_class_id": 0, "port": "command"},
                    "shape": [8],
                    "dtype": "float64",
                }
            ]
        }
    )

    assert control_shms[0]["shm"] == "Robot_0_control"
    assert len(control_shms) == 3
    assert control_shms[1]["shape"] == [1]


@pytest.mark.unit
def test_rollout_controller_uses_configured_task_name() -> None:
    """The rollout saver command should preserve the policy task prompt."""
    owner = SimpleNamespace(
        rollout_python_path="/runtime/python",
        value_shm=FakePolicyOutputShm(),
        action_chunk_shm=FakePolicyOutputShm(),
        q_value_shm=FakePolicyOutputShm(),
        q_value_timestamps_shm=FakePolicyOutputShm(),
        config_file_path=None,
    )
    controller = RolloutController(owner)

    command = controller._build_start_command(
        {"task": "Put gift into box and cover lid"}
    )

    task_index = command.index("--task-name")
    assert command[task_index + 1] == "Put gift into box and cover lid"


@pytest.mark.unit
def test_episode_controller_writes_signal_files(tmp_path) -> None:
    """EpisodeController should own simple episode signal file operations."""
    owner = SimpleNamespace(
        signal_files={
            "episode_result": str(tmp_path / "episode_result"),
            "policy_runner_reset_command": str(tmp_path / "policy_runner_reset"),
            "teleoperator_reset_command": str(tmp_path / "teleoperator_reset"),
            "robot_home_index": str(tmp_path / "robot_home_index"),
        },
    )
    controller = EpisodeController(owner)

    controller.write_episode_result("success")
    controller.signal_policy_runner_reset()
    controller.signal_teleoperator_reset()
    (tmp_path / "robot_home_index").write_text("3")

    assert (tmp_path / "episode_result").read_text() == "success"
    assert (tmp_path / "policy_runner_reset").exists()
    assert (tmp_path / "teleoperator_reset").exists()
    assert controller.read_robot_home_index() == 3


@pytest.mark.unit
def test_episode_controller_consumes_robot_reconnect_signal(tmp_path) -> None:
    """Robot reconnect should enqueue an episode discard and consume the signal file."""
    reconnect_path = tmp_path / "robot_reconnect"
    reconnect_path.touch()
    owner = SimpleNamespace(
        signal_files={"robot_reconnect": str(reconnect_path)},
        _episode_result_queue=queue.Queue(),
    )
    controller = EpisodeController(owner)

    assert controller.check_and_consume_robot_reconnect_signal() is True
    assert reconnect_path.exists() is False
    assert owner._episode_result_queue.get_nowait() == "discard"


@pytest.mark.unit
def test_episode_controller_signals_robot_home_and_reset(tmp_path) -> None:
    """Robot home signals should be written for controls and reset configs."""
    owner = SimpleNamespace(
        controls_config=[_runtime_control_target("RizonRobot", 1)],
        reset_config=[{"device_name": "Teleop", "device_class_id": 0, "home_index": 7}],
        signal_files={
            "robot_home_command_pattern": str(tmp_path / "{device_name}_{device_class_id}_home"),
        },
    )
    controller = EpisodeController(owner)

    controller.signal_robot_home(home_index=3)

    assert (tmp_path / "RizonRobot_1_home").read_text() == "3"
    assert (tmp_path / "Teleop_0_home").read_text() == "7"


@pytest.mark.unit
def test_episode_controller_writes_sampled_home_json_atomically(tmp_path) -> None:
    """Sampled dual-arm targets must use the JSON Home protocol understood by Rizon."""
    owner = SimpleNamespace(
        controls_config=[_runtime_control_target("RizonRobot", 2)],
        reset_config=[],
        signal_files={
            "robot_home_command_pattern": str(
                tmp_path / "{device_name}_{device_class_id}_home"
            ),
        },
    )
    target = {
        "schema_version": 1,
        "source": "home_pose_candidates",
        "config_name": "fixture",
        "config_hash": "a" * 64,
        "arms": {
            "left": {
                "candidate_id": "left-01",
                "joints": [0.1] * 7,
                "collision_checked": False,
            },
            "right": {
                "candidate_id": "right-01",
                "joints": [-0.1] * 7,
                "collision_checked": False,
            },
        },
    }

    controller = EpisodeController(owner)
    assert controller.signal_robot_home(home_target=target) is True

    command_path = tmp_path / "RizonRobot_2_home"
    assert command_path.exists()
    assert json.loads(command_path.read_text()) == target
    assert owner._sampled_home_result_files


@pytest.mark.unit
def test_episode_controller_uses_sampled_home_for_reset_only_rizon_config(tmp_path) -> None:
    owner = SimpleNamespace(
        controls_config=[],
        reset_config=[{"device_name": "RizonRobot", "device_class_id": 2, "home_index": 4}],
        signal_files={
            "robot_home_command_pattern": str(
                tmp_path / "{device_name}_{device_class_id}_home"
            ),
        },
    )
    target = {
        "schema_version": 1,
        "source": "home_pose_candidates",
        "config_name": "fixture",
        "config_hash": "a" * 64,
        "arms": {
            "left": {"candidate_id": "left-01", "joints": [0.1] * 7},
            "right": {"candidate_id": "right-01", "joints": [-0.1] * 7},
        },
    }

    assert EpisodeController(owner).signal_robot_home(home_target=target) is True
    assert json.loads(
        (tmp_path / "RizonRobot_2_home").read_text(encoding="utf-8")
    )["config_hash"] == target["config_hash"]


@pytest.mark.unit
def test_policy_connector_normalizes_single_device_reset_config() -> None:
    """Hydra commonly resolves a one-entry reset block as a mapping."""
    assert PolicyConnector._normalize_device_specs({"device_name": "RizonRobot"}) == [
        {"device_name": "RizonRobot"}
    ]
    assert PolicyConnector._normalize_device_specs({}) == []


@pytest.mark.unit
def test_episode_controller_signals_halt_and_consumes_reply(tmp_path) -> None:
    """Robot halt should touch command files and consume ready reply files."""
    reply_path = tmp_path / "RizonRobot_1_halt_reply"
    reply_path.touch()
    owner = SimpleNamespace(
        halt_and_home_cfg=[{"device_name": "RizonRobot", "device_class_id": 1}],
        signal_files={
            "robot_halt_command_pattern": str(tmp_path / "{device_name}_{device_class_id}_halt"),
            "robot_halt_reply_pattern": str(tmp_path / "{device_name}_{device_class_id}_halt_reply"),
        },
    )
    controller = EpisodeController(owner)

    controller.signal_and_wait_for_robot_halt()

    assert (tmp_path / "RizonRobot_1_halt").exists()
    assert reply_path.exists() is False


@pytest.mark.unit
def test_policy_visualization_detects_robot_state_devices() -> None:
    """PolicyVisualization should centralize robot-state device detection."""
    owner = SimpleNamespace(robot_state_devices={"ConfiguredRobot_9"})
    visualization = PolicyVisualization(owner)

    assert visualization.is_robot_state_device("ConfiguredRobot_9") is True
    assert visualization.is_robot_state_device("RizonRobot_1") is True
    assert visualization.is_robot_state_device("robot_state") is True
    assert visualization.is_robot_state_device("Camera_0") is False
    assert visualization.is_robot_state_device("RoboTwinCamera_0") is False


@pytest.mark.unit
def test_policy_visualization_logs_only_robot_observations() -> None:
    """Observation xyz logging should ignore non-robot device names."""
    logger = FakeActionStateLogger()
    owner = SimpleNamespace(
        robot_state_devices=set(),
        action_state_xyz_logger=logger,
    )
    visualization = PolicyVisualization(owner)

    visualization.maybe_log_robot_state_xyz(
        device_name="Camera_0",
        state=np.array([1.0]),
        timestamp_ns=10,
        summary_frame_timestamp_ns=20,
    )
    visualization.maybe_log_robot_state_xyz(
        device_name="RizonRobot_1",
        state=np.array([2.0]),
        timestamp_ns=30,
        summary_frame_timestamp_ns=40,
    )

    assert len(logger.states) == 1
    assert logger.states[0]["device_name"] == "RizonRobot_1"
    assert logger.states[0]["event_type"] == "observation_xyz"


@pytest.mark.unit
def test_policy_visualization_marks_robotwin_joint14_states() -> None:
    logger = FakeActionStateLogger()
    owner = SimpleNamespace(
        robot_state_devices={"RoboTwinSimRobot_0"},
        action_state_xyz_logger=logger,
    )
    visualization = PolicyVisualization(owner)

    visualization.maybe_log_robot_state_xyz(
        device_name="RoboTwinSimRobot_0",
        state=np.arange(14, dtype=np.float64),
        timestamp_ns=10,
        summary_frame_timestamp_ns=20,
    )

    assert logger.states[0]["vector_layout"] == "joint14"


@pytest.mark.unit
def test_policy_history_manager_resolves_control_shm_name() -> None:
    """PolicyHistoryManager should map device names to configured control SHMs."""
    owner = SimpleNamespace(
        controls_config=[
            _runtime_control_target("RizonRobot", 1),
            _runtime_control_target("Gripper", 0),
        ],
    )
    manager = PolicyHistoryManager(owner)

    assert manager.control_shm_name_for_device("RizonRobot_1") == "RizonRobot_1_control"
    assert manager.control_shm_name_for_device("MissingRobot_0") is None


@pytest.mark.unit
def test_policy_history_manager_attaches_arrival_chunk_timing() -> None:
    """Arrival-anchored history specs should receive action chunk timing metadata."""
    history_spec = {"anchor_mode": "arrival"}
    owner = SimpleNamespace(
        _pending_policy_history_spec=history_spec,
        _policy_history_arrival_wait_start_ns=None,
        action_step_interval_ns=20,
    )
    manager = PolicyHistoryManager(owner)

    manager.attach_action_chunk_timing(
        {"Robot_0": np.zeros((3, 2), dtype=np.float64)},
        first_action_sent_timestamp_ns=100,
    )

    assert history_spec["action_chunk_first_action_sent_timestamp_ns"] == 100
    assert history_spec["action_chunk_last_action_sent_timestamp_ns"] == 140
    assert history_spec["action_chunk_step_interval_ns"] == 20
    assert history_spec["action_chunk_length"] == 3
    assert history_spec["action_chunk_device_names"] == ["Robot_0"]


@pytest.mark.unit
def test_policy_history_manager_summary_candidates_sort_and_limit() -> None:
    """Candidate lookup should sort by timeline timestamp after limiting by distance."""
    frame_sizes = [0]
    shm_size = calculate_summary_shm_size(1, frame_sizes, buffer_size=4)
    buf = bytearray(shm_size)
    buf[:MANAGER_HEADER_SIZE] = pack_manager_header(
        device_count=1,
        buffer_size=4,
        current_frames_count=4,
        write_index=0,
        timestamp=1,
    )
    for frame_index, timestamp_ns in enumerate([900, 980, 1_010, 1_090]):
        offset = get_summary_frame_start_offset(frame_index, 1, frame_sizes)
        buf[offset:offset + SUMMARY_FRAME_HEADER_SIZE] = pack_summary_frame_header(
            timestamp_ns
        )
    owner = SimpleNamespace(
        summary_shm=SimpleNamespace(buf=buf),
        summary_shm_lock=nullcontext(),
        devices={"Camera_0": {"frame_size": 0}},
        find_frame_by_timestamp=lambda timestamp_ns, tolerance_ms: {
            "frame_index": 3,
            "frame_timestamp_ns": 1_020,
            "timestamp_diff_ns": 20,
            "timestamp_diff_ms": 0.02,
        },
    )
    manager = PolicyHistoryManager(owner)

    candidates = manager.summary_frame_candidates(
        1_000,
        window_ms=120.0,
        max_candidates=2,
    )

    assert [candidate["frame_index"] for candidate in candidates] == [1, 2]
    assert [candidate["diff_ns"] for candidate in candidates] == [-20, 10]


@pytest.mark.unit
def test_policy_history_manager_collects_frames_by_indices_logs_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manual frame collection should preserve observation, metadata, and state logs."""
    timestamp_events: list[dict[str, Any]] = []
    action_logger = FakeActionStateLogger()
    owner = SimpleNamespace(
        timestamp_logger=SimpleNamespace(
            log_event=lambda event_name, **kwargs: timestamp_events.append(
                {"event_name": event_name, **kwargs}
            )
        ),
        action_state_xyz_logger=action_logger,
        read_all_device_data=lambda **kwargs: (999, {}),
        devices={"Robot_0": {}, "Camera_0": {}},
        _is_robot_state_device=lambda device_name: device_name == "Robot_0",
        enable_policy_history_video=False,
        camera_devices=[],
    )
    manager = PolicyHistoryManager(owner)
    robot_state = np.arange(22, dtype=np.float64)
    camera_frame = np.ones((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(
        manager,
        "read_summary_frame_by_physical_index",
        lambda frame_index: {
            "frame_index": int(frame_index),
            "frame_timestamp_ns": 1_050,
            "data": {
                "Robot_0": (1_040, robot_state),
                "Camera_0": (1_045, camera_frame),
            },
        },
    )
    history_spec = {"base_timestamp_ns": 1_000}

    observations = manager.collect_history_frames_by_indices(
        [7],
        history_kind="history_image",
        history_spec=history_spec,
        target_specs=[{"step": 2, "target_timestamp_ns": 1_020}],
    )

    assert observations is not None
    assert set(observations[0]) == {"Robot_0", "Camera_0"}
    assert timestamp_events[0]["event_name"] == "history_image_manual_read"
    assert timestamp_events[0]["matched_frame_index"] == 7
    assert history_spec["history_image_frame_infos"][0]["manual_selection"] is True
    assert action_logger.states[0]["event_type"] == "history_image"
    assert action_logger.states[0]["history_state_timestamp_ns"] == 1_040


@pytest.mark.unit
def test_policy_history_manager_writes_manual_picker_html(tmp_path: Any) -> None:
    """HTML generation should choose nearest candidates and include escaped preview paths."""
    manager = PolicyHistoryManager(SimpleNamespace())
    html_path = tmp_path / "picker.html"

    manager.write_history_manual_picker_html(
        html_path=html_path,
        image_groups=[
            {
                "index": 0,
                "step": 2,
                "target_timestamp_ns": 2_000_000_000,
                "candidates": [
                    {
                        "frame_index": 5,
                        "frame_timestamp_ns": 1_999_000_000,
                        "diff_ms": -1.0,
                        "preview": "preview&one.png",
                    },
                    {
                        "frame_index": 6,
                        "frame_timestamp_ns": 2_000_100_000,
                        "diff_ms": 0.1,
                    },
                ],
            }
        ],
        pose_groups=[],
    )

    html_text = html_path.read_text(encoding="utf-8")
    assert "Manual History Picker" in html_text
    assert "Image step 2" in html_text
    assert 'value=\'6\' checked' in html_text
    assert "preview&amp;one.png" in html_text


@pytest.mark.unit
def test_policy_history_manager_updates_policy_history_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """History update orchestration should update policy, log meta, and clear state."""
    class FakeHistoryPolicy:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self._last_kv_cache_update_summary = {
                "trigger_timestamp_ns": 10,
                "end_timestamp_ns": 25,
                "duration_ns": 15,
                "history_frames": 1,
                "pose_frames": 1,
                "success": True,
            }

        def update_history_with_observations(
            self,
            keyframe_observations: list[dict[str, np.ndarray]],
            *,
            pose_observations: list[dict[str, np.ndarray]] | None,
        ) -> None:
            self.calls.append(
                {
                    "keyframe_observations": keyframe_observations,
                    "pose_observations": pose_observations,
                }
            )

    policy = FakeHistoryPolicy()
    logger = FakeActionStateLogger()
    relative_logs: list[dict[str, Any]] = []
    owner = SimpleNamespace(
        _pending_policy_history_spec={
            "base_timestamp_ns": 100,
            "keyframe_after_steps": [1],
            "pose_after_steps": [1],
            "pose_interval_ns": 10,
            "tolerance_ms": 80.0,
        },
        _policy_history_arrival_wait_start_ns=None,
        action_step_interval_ns=10,
        policy=policy,
        history_manual_selection_enabled=False,
        action_state_xyz_logger=logger,
        _maybe_log_policy_history_relative_xyz=lambda spec: relative_logs.append(dict(spec)),
    )
    manager = PolicyHistoryManager(owner)
    keyframes = [{"Camera_0": np.array([1.0])}]
    poses = [{"Robot_0": np.array([2.0])}]
    monkeypatch.setattr(manager, "collect_policy_history_observations", lambda spec, block=True: keyframes)
    monkeypatch.setattr(manager, "collect_policy_history_poses", lambda spec, block=True: poses)
    monkeypatch.setattr(manager, "save_policy_history_keyframe_images", lambda spec, obs: None)

    updated = manager.maybe_update_policy_history(latest_obs_time=200)

    assert updated is True
    assert policy.calls == [{"keyframe_observations": keyframes, "pose_observations": poses}]
    assert owner._pending_policy_history_spec is None
    assert [entry["event_type"] for entry in logger.metas] == [
        "kv_cache_update_start",
        "kv_cache_update_end",
    ]
    assert len(relative_logs) == 2
