"""PolicyConnector refactor characterization tests.

Author: OpenAI Assistant
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import numpy as np
import pytest

from consumer.inference.common import policy_connector as connector_module
from consumer.inference.common.policy_connector import PolicyConnector, _normalize_camera_devices
from consumer.inference.common.episode_controller import EpisodeController
from consumer.inference.common.policy_output_writer import PolicyOutputWriter
from consumer.inference.common.policy_runtime_io import PolicyRuntimeIO
from consumer.inference.common.policy_visualization import PolicyVisualization
from consumer.policy_connector import PolicyConnector as FacadePolicyConnector
from consumer.policy_history import PolicyHistoryManager
from utils.shm_utils import (
    FRAME_HEADER_SIZE,
    get_policy_data_offset,
    unpack_frame_header,
)


@pytest.mark.unit
def test_compatibility_facade_exports_refactored_connector() -> None:
    """Legacy imports resolve to the reorganized implementation."""
    assert FacadePolicyConnector is PolicyConnector
    assert not any("Mixin" in base.__name__ for base in PolicyConnector.__mro__)
    assert PolicyVisualization is not None
    assert PolicyHistoryManager is not None


@pytest.mark.unit
def test_camera_role_mapping_uses_observation_device_names() -> None:
    """Semantic policy camera names are not SHM device names for recorders."""
    assert _normalize_camera_devices(
        {
            "head_camera": "RoboTwinCamera_0",
            "left_camera": "RoboTwinCamera_1",
        }
    ) == ["RoboTwinCamera_0", "RoboTwinCamera_1"]
    assert _normalize_camera_devices("RoboTwinCamera_0") == ["RoboTwinCamera_0"]
    assert _normalize_camera_devices(["RoboTwinCamera_0"]) == ["RoboTwinCamera_0"]


@pytest.mark.unit
def test_prediction_action_index_preserves_rtc_first_iteration_barrier() -> None:
    """RTC feedback is clamped and hidden from the first inference only."""
    connector = PolicyConnector.__new__(PolicyConnector)
    connector.policy = SimpleNamespace(enable_rtc=True)
    connector.runtime_io = PolicyRuntimeIO(connector)
    connector._read_policy_action_index = lambda: -4

    assert connector._get_prediction_action_index(0) is None
    assert connector._get_prediction_action_index(1) == 0

    connector._read_policy_action_index = lambda: 7
    assert connector._get_prediction_action_index(2) == 7

    connector.policy = SimpleNamespace(enable_rtc=False)
    connector._read_policy_action_index = lambda: pytest.fail("non-RTC policy read action_index")
    assert connector._get_prediction_action_index(3) is None


@pytest.mark.unit
def test_action_chunk_frame_uses_real_observation_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every policy frame, including the first, carries its observation timestamp."""
    connector = PolicyConnector.__new__(PolicyConnector)
    frame_size = FRAME_HEADER_SIZE + 2 * np.dtype(np.float64).itemsize
    connector.policy_shm = SimpleNamespace(buf=bytearray(4096))
    connector.policy_devices = [
        {
            "device_name": "robot",
            "shape": (2, 1),
            "frame_size": frame_size,
        }
    ]
    connector.timestamp_logger = SimpleNamespace(log_event=lambda *args, **kwargs: None)
    connector.policy_output_writer = PolicyOutputWriter(
        policy_shm_getter=lambda: connector.policy_shm,
        policy_devices_getter=lambda: connector.policy_devices,
        reconnect_policy_shm=lambda: None,
        timestamp_logger=connector.timestamp_logger,
    )
    monkeypatch.setattr(connector_module.time, "time_ns", lambda: 999_000)

    observation_timestamp_ns = 123_456
    write_timestamp_ns = connector._write_action_chunk_dict(
        {"robot": np.asarray([[1.0], [2.0]], dtype=np.float64)},
        observation_timestamp=observation_timestamp_ns,
    )

    data_offset = get_policy_data_offset(0, 1, [frame_size])
    frame_header = unpack_frame_header(
        connector.policy_shm.buf[data_offset : data_offset + FRAME_HEADER_SIZE]
    )
    assert frame_header["timestamp_ns"] == observation_timestamp_ns
    assert write_timestamp_ns == 999_000


@pytest.mark.unit
def test_rollout_recording_is_gated_by_first_action() -> None:
    """Recording starts once, at the first action boundary."""
    connector = PolicyConnector.__new__(PolicyConnector)
    starts: List[str] = []
    connector._infer_rollout_started = False
    connector._signal_start_recording = lambda: starts.append("start")

    assert connector._start_recording_on_first_action(False) is False
    assert starts == []

    connector._infer_rollout_started = True
    assert connector._start_recording_on_first_action(False) is True
    assert connector._start_recording_on_first_action(True) is True
    assert starts == ["start"]


@pytest.mark.unit
def test_episode_transition_keeps_dual_reset_wait_barriers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Policy command buffers are reset and acknowledged both before and after homing."""
    connector = PolicyConnector.__new__(PolicyConnector)
    events: List[Any] = []
    connector._episode_in_transition = False
    connector._episode_count = 4
    connector.episode_halt_settle_time = 0.0
    connector.episode_home_wait_time = 0.0
    connector.episode_auto_restart = True
    connector.inference_enabled = True
    connector._write_episode_result = lambda result: events.append(("result", result))
    connector._signal_stop_recording = lambda: events.append("stop")
    connector._signal_and_wait_for_robot_halt = lambda: events.append("halt")
    connector._signal_policy_runner_reset = lambda: events.append("reset")
    connector._wait_for_policy_runner_reset = lambda: events.append("wait") or True
    connector._reset_policy_episode_state = lambda: events.append("policy_reset")
    connector._read_robot_home_index = lambda: 3
    connector._signal_robot_home = lambda home_index: events.append(("home", home_index))
    connector._signal_teleoperator_reset = lambda: events.append("teleoperator")
    connector._signal_start_recording = lambda: events.append("start")
    connector._last_continuous_state_timestamps = {}
    connector.policy = SimpleNamespace()
    connector.policy_history_manager = SimpleNamespace(clear_state=lambda: events.append("policy_reset"))
    connector.episode_policy_runner_reset_timeout = 1.0
    connector.episode_controller = EpisodeController(connector)
    connector.episode_controller.write_episode_result = connector._write_episode_result
    connector.episode_controller.signal_and_wait_for_robot_halt = connector._signal_and_wait_for_robot_halt
    connector.episode_controller.signal_policy_runner_reset = connector._signal_policy_runner_reset
    connector.episode_controller.wait_for_policy_runner_reset = connector._wait_for_policy_runner_reset
    connector.episode_controller.reset_policy_episode_state = connector._reset_policy_episode_state
    connector.episode_controller.read_robot_home_index = connector._read_robot_home_index
    connector.episode_controller.signal_robot_home = connector._signal_robot_home
    connector.episode_controller.signal_teleoperator_reset = connector._signal_teleoperator_reset
    monkeypatch.setattr(connector_module.time, "sleep", lambda _: None)

    connector._handle_episode_transition("success")

    assert events.count("reset") == 2
    assert events.count("wait") == 2
    assert events.count("policy_reset") == 2
    first_reset = events.index("reset")
    home = events.index(("home", 3))
    second_reset = events.index("reset", first_reset + 1)
    teleoperator = events.index("teleoperator")
    assert first_reset < home < second_reset < teleoperator
    assert events[-1] == "start"
    assert connector.inference_enabled is True
    assert connector._episode_count == 5
    assert connector._episode_in_transition is False
