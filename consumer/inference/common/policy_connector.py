#!/usr/bin/env python3
"""
Policy Connector - runs a policy and writes action chunking to policy SHM.

This module:
1. Reads observations from summary SHM (via BaseConsumer)
2. Runs a policy to compute action chunking
3. Writes action chunking into policy SHM for ActionExecutor to consume

Policy SHM is connect-only, created by ActionExecutor.

Author: Jun Lv, Han Xue, Zheng Wang
"""

import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from consumer.core.base import BaseConsumer
from consumer.inference.common.episode_controller import EpisodeController
from consumer.inference.common.observation_builder import ObservationBuilder
from consumer.inference.common.policy_loop import PolicyLoop
from consumer.inference.common.policy_output_writer import PolicyOutputWriter
from consumer.inference.common.policy_runtime import PolicyRuntime
from consumer.inference.common.policy_runtime_io import PolicyRuntimeIO
from consumer.inference.common.policy_visualization import PolicyVisualization
from consumer.inference.common.rollout_controller import RolloutController
from consumer.inference.common.value_prediction import ValuePredictionWorker
from consumer.policy_history import PolicyHistoryManager
from utils.action_state_xyz_logger import ActionStateXYZLogger
from utils.inference_profiling import InferenceProfiler
from utils.logger_config import logger
from utils.observation_recorder import TimestampedObservationRecorder
from utils.policy_control_config import parse_control_targets
from utils.runtime_profiles import select_runtime_python
from utils.shm_utils import PolicyOutputSHM
from utils.timestamp_event_logger import TimestampEventLogger

try:
    # This module is intentionally dependency-free, so the inference worker can
    # consume the control-panel spool without importing Flask or hardware code.
    from utils.inference_control import InferenceControlConsumer
    from utils.recording_control import WholeEpisodeCommandConsumer
except (ImportError, OSError):  # Native launches do not set the control-panel env.
    InferenceControlConsumer = None  # type: ignore[assignment,misc]
    WholeEpisodeCommandConsumer = None  # type: ignore[assignment,misc]

ActionType = Union[np.ndarray, Dict[str, np.ndarray]]
_FASTWAM_POLICY_CLASS = "FastWAMPolicy"


def _build_policy_constructor_kwargs(
    policy_class: str,
    policy_params: Optional[Dict[str, Any]],
    profiler: Optional[InferenceProfiler],
) -> Dict[str, Any]:
    """Inject FastWAM-only runtime helpers without changing other policies."""
    kwargs = dict(policy_params or {})
    if policy_class == _FASTWAM_POLICY_CLASS:
        kwargs.setdefault("profiler", profiler)
    return kwargs


def _resolve_action_chunk_observation_timestamp(
    policy_class: str,
    observation_timestamp_ns: int,
    iteration_count: int,
    *,
    current_time_ns: Optional[int] = None,
) -> int:
    """Keep source observation timestamps for FastWAM and later chunks."""
    if policy_class == _FASTWAM_POLICY_CLASS or iteration_count > 0:
        return int(observation_timestamp_ns)
    return int(time.time_ns() if current_time_ns is None else current_time_ns)


def _make_run_timestamp() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}_pid{os.getpid()}"


def _normalize_camera_devices(raw: Any) -> List[str]:
    """Normalize list-style devices and semantic camera-role mappings."""
    if isinstance(raw, Mapping):
        return [str(device_name) for device_name in raw.values()]
    if isinstance(raw, str):
        return [raw]
    return [str(device_name) for device_name in (raw or [])]


def _resolve_rollout_python_path(
    infer_rollout_config: Dict[str, Any],
    environ: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Resolve Control Panel, local-profile, config, then current recorder Python."""
    environment = os.environ if environ is None else environ
    configured_path = infer_rollout_config.get("python_path")
    fallback = (
        configured_path
        if isinstance(configured_path, str) and configured_path.strip()
        else sys.executable
    )
    selection = select_runtime_python(
        "recording",
        environment_names=(
            "SUPERINFERENCE_RECORDER_PYTHON",
            "SUPERINFERENCE_RECORDING_PYTHON",
        ),
        config_name=environment.get("SUPERINFERENCE_CONFIG_NAME"),
        fallback=fallback,
        environ=environment,
        require_verified_profile=True,
    )
    return str(selection.python) if selection is not None else None


class PolicyConnector(BaseConsumer):
    """
    Connector that:
      - Reads observations from summary SHM (via BaseConsumer)
      - Runs a policy to compute action chunking
      - Writes action chunking into policy SHM for ActionExecutor to consume

    Policy SHM is connect-only, created by ActionExecutor.
    """

    def __init__(self,
                 summary_shm_name: str = "device_summary_data",
                 policy_class: str = "TinyMLPPolicy",
                 policy_params: Optional[Dict[str, Any]] = None,
                 obs_devices: Optional[List[str]] = None,
                 controls: Optional[List[Dict[str, Any]]] = None,
                 reset_cfg: Optional[List[Dict[str, Any]]] = None,
                 halt_and_home_cfg: Optional[List[Dict[str, Any]]] = None,
                 fps: float = 50.0,
                 control_fps: float = 50.0,
                 value_prediction_query_fps: float = 100.0,
                 history_with_value: bool = False,
                 enable_value_prediction: bool = False,
                 chunk_length: int = 8,
                 retry_connect_secs: float = 0.5,
                 master_device: Optional[str] = None,
                 infer_rollout_config: Optional[Dict[str, Any]] = None,
                 policy_history_video_config: Optional[Dict[str, Any]] = None,
                 config_file_path: Optional[str] = None,
                 runtime_config: Optional[Dict[str, Any]] = None,
                 observation_recorder_config: Optional[Dict[str, Any]] = None,
                 profiling_config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(summary_shm_name)

        # Store config file path for rollout saving
        self.config_file_path = config_file_path
        self.policy_class = policy_class
        self.policy_params = dict(policy_params or {})
        self.profiler: Optional[InferenceProfiler] = (
            InferenceProfiler(profiling_config, component="client")
            if self.policy_class == _FASTWAM_POLICY_CLASS
            else None
        )
        # Some multimodal policies use semantic camera-role mappings such as
        # {"head_camera": "RoboTwinCamera_0"}. Recorders need actual
        # observation device names rather than those role keys.
        self.camera_devices = _normalize_camera_devices(self.policy_params.get("camera_devices", []))
        self.obs_devices = obs_devices or []
        assert master_device is None or (
            master_device in self.obs_devices
        ), f"master_device is not in obs_devices {self.obs_devices}"
        self.master_device = master_device
        self.chunk_length = chunk_length

        # Resolve control endpoints once. Raw SHM names are intentionally not
        # accepted at this boundary; every runtime user consumes these derived
        # canonical fields instead.
        self.controls_config: List[Dict[str, Any]] = [
            target.runtime_mapping()
            for target in parse_control_targets(controls, field="policy.controls")
        ]

        # Build reset config to signal device to reset whenever an episode is switched
        self.reset_config = self._normalize_device_specs(reset_cfg)
        self.halt_and_home_cfg = self._normalize_device_specs(halt_and_home_cfg)

        self.fps = fps
        self.control_fps = control_fps
        self.history_with_value = history_with_value
        self.value_prediction_query_fps = value_prediction_query_fps
        self.update_interval = 1.0 / max(self.fps, 1e-6)
        self.value_prediction_query_interval = 1.0 / max(self.value_prediction_query_fps, 1e-6)
        self.retry_connect_secs = retry_connect_secs
        self.enable_value_prediction = enable_value_prediction

        # Policy SHM for outputting action chunking
        self.policy_shm = None  # Will be Union[shm.SharedMemory, ReadOnlySharedMemory]
        self.runtime_io = PolicyRuntimeIO(self)

        # Build policy through a dedicated runtime wrapper while keeping self.policy
        # for existing methods and subclasses.
        self.policy_runtime = PolicyRuntime(
            self.policy_class,
            _build_policy_constructor_kwargs(
                self.policy_class,
                self.policy_params,
                self.profiler,
            ),
        )
        self.policy = self.policy_runtime.policy
        configured_robot_device = self.policy_params.get("robot_device")
        recorder_config = dict(observation_recorder_config or {})
        recorder_config.setdefault("enabled", False)
        self.observation_recorder = TimestampedObservationRecorder(
            recorder_config,
            run_started_at_ns=getattr(self.policy, "_inference_run_started_at_ns", time.time_ns()),
            device_names=self.obs_devices,
            camera_devices=recorder_config.get("camera_devices", self.camera_devices),
            robot_device=recorder_config.get(
                "robot_device",
                str(configured_robot_device) if configured_robot_device else None,
            ),
        )
        self.observation_recorder_dir: Optional[Path] = None
        self._observation_timestamp_metadata: Dict[int, Tuple[int, Dict[str, int]]] = {}
        self._observation_timestamp_metadata_lock = threading.Lock()
        self.timestamp_logger = TimestampEventLogger("policy_connector")
        self.action_state_xyz_logger = ActionStateXYZLogger("policy_connector")
        configured_robot_device = self.policy_params.get("robot_device")
        self.robot_state_devices = {
            str(configured_robot_device)
        } if configured_robot_device else set()
        self.continuous_state_fps = float(
            self.policy_params.get(
                "continuous_state_fps",
                os.environ.get("SI_CONTINUOUS_STATE_XYZ_FPS", str(max(self.control_fps, self.fps))),
            )
        )
        self.continuous_state_thread: Optional[threading.Thread] = None
        self.continuous_state_running = False
        self._last_continuous_state_timestamps: Dict[str, int] = {}
        # Action step interval: 1/control_fps seconds (control_fps Hz)
        self.action_step_interval_s: float = 1.0 / self.control_fps
        self.action_step_interval_ns: int = int(self.action_step_interval_s * 1e9)
        self._inference_count = 0
        self._first_inference_start_timestamp_ns: Optional[int] = None
        self._last_inference_start_timestamp_ns: Optional[int] = None

        # Cache control metadata for policy prediction
        self.action_configs: List[Dict[str, Any]] = self._build_action_configs()

        # Inference rollout configuration
        self.infer_rollout_config: Dict[str, Any] = dict(infer_rollout_config or {})
        self.infer_rollout_process: Optional[subprocess.Popen] = None
        self._infer_rollout_started: bool = False

        # Policy history configuration
        self.policy_history_video_config: Dict[str, Any] = dict(
            policy_history_video_config or {}
        )
        self.enable_policy_history_video: bool = bool(
            self.policy_history_video_config.get("enabled", False)
        )
        self.policy_history_video_fps: float = float(
            self.policy_history_video_config.get("fps", 10.0)
        )
        history_video_dir = str(
            self.policy_history_video_config.get(
                "save_dir", "/tmp/superinference_policy_history_videos"
            )
        )
        policy_history_video_base_dir = Path(history_video_dir).expanduser()
        self.policy_history_video_dir: Path = policy_history_video_base_dir / _make_run_timestamp()
        self.policy_history_video_all_dir: Path = self.policy_history_video_dir / "all"
        self._policy_history_video_clip_index: int = 0
        self._policy_history_video_all_counters: Dict[str, int] = {}
        if self.enable_policy_history_video:
            self.policy_history_video_dir.mkdir(parents=True, exist_ok=True)
            self.policy_history_video_all_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "PolicyConnector: History keyframe image saving enabled at "
                f"{self.policy_history_video_dir}"
            )
        self.history_manual_selection_config: Dict[str, Any] = dict(
            self.policy_params.get("history_manual_selection", {}) or {}
        )
        self.history_manual_selection_enabled: bool = bool(
            self.history_manual_selection_config.get("enabled", False)
        )
        self.history_manual_selection_window_ms: float = float(
            self.history_manual_selection_config.get("window_ms", 500.0)
        )
        self.history_manual_selection_max_candidates: int = int(
            self.history_manual_selection_config.get("max_candidates_per_step", 15)
        )
        self.history_manual_selection_timeout_s: float = float(
            self.history_manual_selection_config.get("timeout_s", 0.0)
        )
        self.history_manual_selection_auto_open: bool = bool(
            self.history_manual_selection_config.get("auto_open_browser", True)
        )
        self.history_manual_selection_port: int = int(
            self.history_manual_selection_config.get("port", 0)
        )
        manual_dir = str(
            self.history_manual_selection_config.get(
                "save_dir", "/tmp/superinference_manual_history_picker"
            )
        )
        self.history_manual_selection_dir: Path = (
            Path(manual_dir).expanduser() / _make_run_timestamp()
        )
        self._history_manual_selection_index: int = 0
        if self.history_manual_selection_enabled:
            self.history_manual_selection_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "PolicyConnector: Manual history selection enabled at "
                f"{self.history_manual_selection_dir}"
            )
        self._pending_policy_history_spec: Optional[Dict[str, Any]] = None
        self._policy_history_arrival_wait_start_ns: Optional[int] = None
        self._policy_history_control_shms: Dict[str, Any] = {}

        # Control Panel can supply a dedicated NEDF3 interpreter; native launches
        # default to the current interpreter unless config/env overrides it.
        self.rollout_python_path = _resolve_rollout_python_path(
            self.infer_rollout_config
        )

        # Extract signal files from runtime config
        signal_files = {}
        if runtime_config is not None:
            signal_files = runtime_config.get('signal_files', {})

        # Validate required signal files
        required_signals = ['start_recording', 'stop_recording', 'reset_signal', 'robot_home_index',
                           'episode_result', 'policy_runner_reset_command', 'teleoperator_reset_command',
                           'rollout_ready', 'robot_reconnect', 'robot_halt_command_pattern', 'robot_halt_reply_pattern']
        for sig in required_signals:
            if sig not in signal_files:
                raise ValueError(f"signal_files.{sig} is required in runtime config")

        self.signal_files = signal_files

        # Control-panel pause/reset commands are delivered through a private
        # filesystem spool.  Native launches remain unchanged when the
        # environment variables are absent.
        self.inference_control_consumer = (
            InferenceControlConsumer.from_environment()
            if InferenceControlConsumer is not None
            else None
        )
        self.recording_episode_consumer = (
            WholeEpisodeCommandConsumer.from_environment()
            if WholeEpisodeCommandConsumer is not None
            else None
        )
        self._standalone_reset_pending = False
        self._standalone_reset_command_id: Optional[str] = None

        # Default to enabled when starting
        self.inference_enabled = True

        # Reset signal file path - used to trigger SHM reconnection after device_starter restart
        self.last_reset_signal_time = 0.0

        # Episode control configuration
        episode_control_config = self.infer_rollout_config.get('episode_control', {})
        self.episode_control_enabled = episode_control_config.get('enabled', False)
        self.episode_success_key = episode_control_config.get('success_key', 's')
        self.episode_failure_key = episode_control_config.get('failure_key', 'f')
        self.episode_discard_key = episode_control_config.get('discard_key', 'd')
        self.episode_auto_restart = episode_control_config.get('auto_restart', True)
        self.episode_home_wait_time = episode_control_config.get('home_wait_time', 3.0)
        self.episode_halt_settle_time = episode_control_config.get('halt_settle_time', 0.5)
        self.episode_policy_runner_reset_timeout = episode_control_config.get(
            'policy_runner_reset_timeout', 1.0
        )

        # Episode control state
        self._episode_result_queue: queue.Queue = queue.Queue()
        self._keyboard_listener = None
        self._episode_in_transition = False
        self._episode_count = 0

        # Policy outputs SHM - initialized from config
        self.policy_outputs_config: Dict[str, Any] = dict(self.infer_rollout_config.get('policy_outputs', {}))

        # Value SHM for storing value estimation from policy
        self.value_shm: PolicyOutputSHM = PolicyOutputSHM(
            config=self.policy_outputs_config.get('value', {}),
            fps=self.fps
        )

        # Action chunk SHM for recording predicted chunks
        self.action_chunk_shm: PolicyOutputSHM = PolicyOutputSHM(
            config=self.policy_outputs_config.get('action_chunk', {}),
            fps=self.fps
        )

        # Q-value SHM for storing Q-function values (one per action in chunk)
        self.q_value_shm: PolicyOutputSHM = PolicyOutputSHM(
            config=self.policy_outputs_config.get('q_value', {}),
            fps=self.fps
        )

        # Q-value timestamps SHM for storing per-action timestamps
        self.q_value_timestamps_shm: PolicyOutputSHM = PolicyOutputSHM(
            config=self.policy_outputs_config.get('q_value_timestamps', {}),
            fps=self.fps
        )

        # Rollout-only SHMs: skip create/write when infer_rollout is disabled
        if not self.infer_rollout_config.get('enabled', False):
            for output_shm in (
                self.value_shm,
                self.action_chunk_shm,
                self.q_value_shm,
                self.q_value_timestamps_shm,
            ):
                output_shm.enabled = False

        # Value prediction thread
        # self.value_prediction_queue: queue.Queue = queue.Queue(maxsize=10)
        self.value_prediction_thread: Optional[threading.Thread] = None
        self.value_prediction_running: bool = False

        # Component boundaries for the first behavior-preserving PolicyConnector split.
        self.policy_visualization = PolicyVisualization(self)
        self.observation_builder = ObservationBuilder(
            read_all_device_data=self.read_all_device_data,
            master_device=self.master_device,
            robot_state_logger=self.policy_visualization.maybe_log_robot_state_xyz,
            owner=self,
        )
        self.policy_output_writer = PolicyOutputWriter(
            policy_shm_getter=lambda: self.policy_shm,
            policy_devices_getter=lambda: getattr(self, "policy_devices", []),
            reconnect_policy_shm=self._connect_policy_shm,
            timestamp_logger=self.timestamp_logger,
        )
        self.episode_controller = EpisodeController(self)
        self.value_prediction_worker = ValuePredictionWorker(self)
        self.rollout_controller = RolloutController(self)
        self.policy_history_manager = PolicyHistoryManager(self)
        self.policy_loop = PolicyLoop(self)

    @staticmethod
    def _normalize_device_specs(raw: Any) -> List[Dict[str, Any]]:
        """Accept one device mapping as well as the historical list form."""
        if raw is None:
            return []
        if isinstance(raw, Mapping):
            return [dict(raw)] if raw else []
        if isinstance(raw, (list, tuple)):
            return [dict(item) for item in raw if isinstance(item, Mapping)]
        return []

    def _apply_inference_control_command(self, command: Dict[str, Any]) -> None:
        """Apply one panel command in the inference worker process."""
        action = command.get("action")
        if action == "pause":
            self.inference_enabled = False
            self.policy_history_manager.clear_state()
            if self.halt_and_home_cfg:
                self._signal_robot_halt()
            # ActionExecutor owns the action queue.  Its existing reset signal
            # clears queued/current actions without unloading the model.
            self._signal_policy_runner_reset()
            logger.info("PolicyConnector: Control-panel pause applied")
            return

        if action == "resume":
            self.inference_enabled = True
            logger.info("PolicyConnector: Control-panel resume applied")
            return

        if action == "reset":
            self.inference_enabled = False
            self._standalone_reset_pending = True
            self._standalone_reset_command_id = str(command.get("command_id") or "")
            self.policy_history_manager.clear_state()
            # Halt is optional in older configs. When configured, signal it so the
            # robot can stop before Home picks up the command. The panel command
            # must remain non-blocking; the reset signal acknowledges completion.
            if self.halt_and_home_cfg:
                self._signal_robot_halt()
            self._signal_policy_runner_reset()
            if getattr(self, "controls_config", None) or getattr(self, "reset_config", None):
                threading.Thread(
                    target=self._complete_standalone_reset,
                    kwargs={"home_target": command.get("home_target")},
                    name="ControlPanelResetHome",
                    daemon=True,
                ).start()
            else:
                # There is no robot command channel in software-only configs;
                # the reset barrier is the only completion event available.
                self._standalone_reset_pending = False
                self.inference_enabled = True
                self._standalone_reset_command_id = None
            logger.info("PolicyConnector: Control-panel reset applied")
            return

        raise ValueError(f"Unsupported inference control action: {action}")

    def _process_inference_control_commands(self) -> None:
        consumer = self.inference_control_consumer
        if consumer is not None:
            consumer.process_commands(
                self._apply_inference_control_command,
                state_provider=lambda: {
                    "inference_paused": not bool(self.inference_enabled),
                    "reset_pending": bool(self._standalone_reset_pending),
                },
            )
        self._process_recording_episode_commands()

    def _process_recording_episode_commands(self) -> None:
        """Consume Control Panel whole-episode commands in the policy owner."""
        consumer = self.recording_episode_consumer
        if consumer is None:
            return

        def handle(command: Dict[str, Any]) -> None:
            action = command.get("action")
            result = "success" if action == "episode_accept" else "discard"
            previous_episode_id = str(command.get("episode_id") or "")
            self.episode_controller.handle_episode_transition(result)
            consumer.wait_for_next_episode(
                previous_episode_id,
                auto_restart=bool(self.episode_auto_restart),
                timeout_s=max(30.0, float(self.episode_home_wait_time) + 15.0),
            )

        consumer.process_commands(handle)

    def _complete_standalone_reset(
        self, *, home_target: Optional[Dict[str, Any]] = None
    ) -> None:
        """Wait for queued actions to clear, then request the configured Home pose."""
        command_id = self._standalone_reset_command_id
        consumer = self.inference_control_consumer
        try:
            if not self._wait_for_policy_runner_reset():
                logger.warning(
                    "PolicyConnector: Timed out waiting for policy runner reset; "
                    "continuing with Home command"
                )
            if not self._signal_robot_home(home_target=home_target):
                raise RuntimeError("No robot Home command could be written")
            if home_target is not None and getattr(self, "_sampled_home_result_files", None):
                timeout_s = float(getattr(self, "episode_home_wait_time", 30.0))
                if not self._wait_for_sampled_home_results(home_target, timeout_s=timeout_s):
                    raise RuntimeError("Sampled robot Home transition failed")
        except Exception as exc:
            logger.warning("PolicyConnector: Standalone reset Home request failed: {}", exc)
            self._standalone_reset_pending = False
            if command_id and consumer is not None:
                try:
                    consumer.update_applied_state(
                        command_id,
                        inference_paused=True,
                        reset_pending=False,
                        status="failed",
                        error={
                            "code": "reset_home_failed",
                            "message": f"{type(exc).__name__}: {exc}",
                        },
                    )
                except Exception as update_exc:
                    logger.warning(
                        "PolicyConnector: Failed to publish reset failure state: {}",
                        update_exc,
                    )

    def _check_reset_signal(self) -> bool:
        """Check if reset signal file exists and has been updated."""
        return self.episode_controller.check_reset_signal()

    def find_value_by_timestamp_with_history_too_old_detection(
        self, target_timestamp_ns: int, tolerance_ms: float = 50.0
    ) -> Optional[Union[np.ndarray, str]]:
        return self.runtime_io.find_value_by_timestamp_with_history_too_old_detection(
            target_timestamp_ns,
            tolerance_ms,
        )

    def _check_and_consume_robot_reconnect_signal(self) -> bool:
        """
        Check and consume robot reconnect signal, then enqueue episode discard.

        Returns:
            True if reconnect signal is detected and consumed, otherwise False
        """
        return self.episode_controller.check_and_consume_robot_reconnect_signal()

    def _value_prediction_worker(self) -> None:
        """Background thread worker for value prediction."""
        self.value_prediction_worker.run()


    def _reconnect_all_shms(self) -> None:
        self.runtime_io._reconnect_all_shms()


    def _merge_action_chunks(
        self, action_chunk_dict: Dict[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        return self.runtime_io._merge_action_chunks(action_chunk_dict)


    def _connect_policy_shm(self) -> None:
        self.runtime_io._connect_policy_shm()


    def _read_policy_action_index(self) -> Optional[int]:
        return self.runtime_io._read_policy_action_index()


    def _get_prediction_action_index(self, iteration_count: int) -> Optional[int]:
        return self.runtime_io._get_prediction_action_index(
            iteration_count,
            read_action_index=self._read_policy_action_index,
        )


    def _build_action_configs(self) -> List[Dict[str, Any]]:
        return self.runtime_io._build_action_configs()


    def _get_action_chunk_shape(self) -> tuple:
        return self.runtime_io._get_action_chunk_shape()

    # Recorder management -------------------------------------------------

    def _should_start_infer_rollout(self) -> bool:
        """Check if inference rollout should be started."""
        return self.rollout_controller.should_start()

    def _start_infer_rollout(self) -> None:
        """Start the inference rollout data saver process using subprocess."""
        self.rollout_controller.start()

    def _stop_infer_rollout(self) -> None:
        """Stop the inference rollout data saver subprocess."""
        self.rollout_controller.stop()

    def _signal_start_recording(self) -> None:
        """Signal the rollout saver to start recording."""
        self.rollout_controller.signal_start_recording()

    def _signal_stop_recording(self) -> None:
        """Signal the rollout saver to stop recording."""
        self.rollout_controller.signal_stop_recording()

    def _start_recording_on_first_action(self, first_action_written: bool) -> bool:
        """Start rollout recording once, immediately before the first action write."""
        if self._infer_rollout_started and not first_action_written:
            self._signal_start_recording()
            return True
        return first_action_written

    # ==================== Episode Control Methods ====================

    def _setup_keyboard_listener(self) -> None:
        """Setup non-blocking keyboard listener for episode control."""
        if not self.episode_control_enabled:
            return

        try:
            from pynput import keyboard as pynput_keyboard

            def on_press(key: Any) -> None:
                try:
                    if hasattr(key, 'char') and key.char:
                        char = key.char.lower()
                        if char == self.episode_success_key:
                            self._episode_result_queue.put('success')
                            logger.info("PolicyConnector: 'S' pressed - marking episode as SUCCESS")
                        elif char == self.episode_failure_key:
                            self._episode_result_queue.put('failure')
                            logger.info("PolicyConnector: 'F' pressed - marking episode as FAILURE")
                        elif char == self.episode_discard_key:
                            self._episode_result_queue.put('discard')
                            logger.info("PolicyConnector: 'D' pressed - discarding episode data")
                except AttributeError:
                    pass

            self._keyboard_listener = pynput_keyboard.Listener(on_press=on_press)
            self._keyboard_listener.start()
            logger.info(f"PolicyConnector: Keyboard listener started. Press '{self.episode_success_key.upper()}' for success, "
                       f"'{self.episode_failure_key.upper()}' for failure, '{self.episode_discard_key.upper()}' to discard episode data.")
        except Exception as e:
            logger.error(f"PolicyConnector: Failed to setup keyboard listener: {e}")
            self._keyboard_listener = None

    def _stop_keyboard_listener(self) -> None:
        """Stop the keyboard listener."""
        if self._keyboard_listener is not None:
            try:
                self._keyboard_listener.stop()
                self._keyboard_listener = None
                logger.info("PolicyConnector: Keyboard listener stopped")
            except Exception as e:
                logger.warning(f"PolicyConnector: Failed to stop keyboard listener: {e}")

    def _write_episode_result(self, result: str) -> None:
        """
        Write episode result to signal file for data saver to read.

        Args:
            result: 'success', 'failure', or 'discard'
        """
        self.episode_controller.write_episode_result(result)

    def _signal_policy_runner_reset(self) -> None:
        """Signal the policy runner to clear its action cache and SHM buffers."""
        self.episode_controller.signal_policy_runner_reset()

    def _wait_for_policy_runner_reset(self, timeout_s: Optional[float] = None) -> bool:
        return self.episode_controller.wait_for_policy_runner_reset(timeout_s)

    def _reset_policy_episode_state(self) -> None:
        self.episode_controller.reset_policy_episode_state()

    def _signal_teleoperator_reset(self) -> None:
        """
        Signal the teleoperator to switch to policy control mode.

        This is called at the start of a new episode to ensure policy control
        is active regardless of how the previous episode ended (human or policy).
        """
        self.episode_controller.signal_teleoperator_reset()

    def _read_robot_home_index(self) -> int:
        """
        Read robot home index from shared file written by device_starter.py.

        Returns:
            Home preset index (default 0 if file not found or invalid)
        """
        return self.episode_controller.read_robot_home_index()

    def _signal_robot_home(
        self,
        home_index: int = 0,
        home_target: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Signal all robots in controls config to return to home pose.

        Args:
            home_index: Home preset index to use (default 0)
        """
        return self.episode_controller.signal_robot_home(
            home_index=home_index,
            home_target=home_target,
        )

    def _wait_for_sampled_home_results(
        self,
        home_target: Dict[str, Any],
        *,
        timeout_s: float,
    ) -> bool:
        return self.episode_controller.wait_for_sampled_home_results(
            home_target,
            timeout_s=timeout_s,
        )

    def _signal_and_wait_for_robot_halt(self) -> None:
        """Signal the robot to halt and wait for the robot to report its "do halt state"."""
        self.episode_controller.signal_and_wait_for_robot_halt()

    def _signal_robot_halt(self) -> None:
        """Signal the robot to halt without blocking the policy loop."""
        self.episode_controller.signal_robot_halt()


    def _handle_episode_transition(self, result: str) -> None:
        """
        Handle episode end: save result, reset, return home, and optionally start new episode.

        Args:
            result: 'success', 'failure', or 'discard'
        """
        self.episode_controller.handle_episode_transition(result)

    def _check_episode_end_request(self) -> Optional[str]:
        """
        Check if there's an episode end request in the queue.

        Returns:
            'success', 'failure', 'discard', or None if no request
        """
        return self.episode_controller.check_episode_end_request()

    # ==================== End Episode Control Methods ====================

    def _requires_history(self) -> bool:
        return self.observation_builder._requires_history()

    def _collect_observation_history(
        self,
        latest_obs: Dict[str, np.ndarray],
        latest_timestamp: int,
        history_with_value: bool = False,
    ) -> Optional[Dict[str, List[np.ndarray]]]:
        return self.observation_builder._collect_observation_history(
            latest_obs,
            latest_timestamp,
            history_with_value=history_with_value,
        )

    def _collect_observation_history_unlocked(
        self,
        latest_obs: Dict[str, np.ndarray],
        latest_timestamp: int,
        history_with_value: bool = False,
    ) -> Optional[Dict[str, List[np.ndarray]]]:
        return self.observation_builder._collect_observation_history_unlocked(
            latest_obs,
            latest_timestamp,
            history_with_value=history_with_value,
        )

    def _prepare_policy_input(
        self,
        latest_obs: Dict[str, np.ndarray],
        latest_timestamp: int,
        history_with_value: bool = False,
    ) -> Optional[Dict[str, Any]]:
        return self.observation_builder._prepare_policy_input(
            latest_obs,
            latest_timestamp,
            history_with_value=history_with_value,
        )

    def _write_action_chunk_dict(
        self,
        action_chunk_dict: Dict[str, np.ndarray],
        observation_timestamp: int = -1,
    ) -> Optional[int]:
        """Write dict-based action chunking to policy SHM."""
        return self.policy_output_writer.write_action_chunk_dict(
            action_chunk_dict,
            observation_timestamp=observation_timestamp,
        )

    def _assemble_observation_dict(self) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[int]]:
        """Assemble observation dictionary using atomic read operation."""
        result = self.observation_builder.assemble()
        return result

    def _read_observation_recorder_frame(self) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Read one synchronized summary frame for the optional recorder."""
        device_names = list(self.obs_devices or getattr(self, "devices", {}).keys())
        if not device_names:
            return None
        result = self.read_all_device_data(
            include_frame_timestamp=True,
            device_names=device_names,
        )
        if result is None:
            return None
        frame_timestamp_ns, all_data = result
        if not all_data or any(name not in all_data for name in device_names):
            return None
        return int(frame_timestamp_ns), all_data

    def _record_observation_metadata(
        self,
        observation_timestamp_ns: int,
        frame_timestamp_ns: int,
        all_data: Dict[str, Tuple[int, np.ndarray]],
    ) -> None:
        """Cache timestamp correspondence for exact recorder audit rows."""
        with self._observation_timestamp_metadata_lock:
            self._observation_timestamp_metadata[int(observation_timestamp_ns)] = (
                int(frame_timestamp_ns),
                {str(name): int(item[0]) for name, item in all_data.items()},
            )
            while len(self._observation_timestamp_metadata) > 256:
                oldest = next(iter(self._observation_timestamp_metadata))
                self._observation_timestamp_metadata.pop(oldest, None)

    def _record_exact_policy_input(
        self,
        policy_input: Dict[str, Any],
        observation_timestamp_ns: int,
        *,
        action_index: Optional[int],
        inference_start_timestamp_ns: int,
    ) -> None:
        """Persist the exact policy input when recording is enabled."""
        recorder = getattr(self, "observation_recorder", None)
        if recorder is None or not getattr(recorder, "enabled", False):
            return
        try:
            with self._observation_timestamp_metadata_lock:
                metadata = self._observation_timestamp_metadata.get(
                    int(observation_timestamp_ns)
                )
            recorder.record_inference(
                int(observation_timestamp_ns),
                policy_input,
                action_index=action_index,
                inference_start_timestamp_ns=int(inference_start_timestamp_ns),
                summary_timestamp_ns=None if metadata is None else metadata[0],
                device_timestamps_ns=None if metadata is None else metadata[1],
                episode_index=getattr(self, "_episode_count", None),
                episode_start_timestamp_ns=getattr(
                    self, "_history_episode_start_timestamp_ns", None
                ),
            )
        except Exception as exc:
            logger.warning(f"PolicyConnector: failed to record exact policy input: {exc}")

    def _maybe_log_robot_state_xyz(
        self,
        *,
        device_name: str,
        state: np.ndarray,
        timestamp_ns: int,
        summary_frame_timestamp_ns: int,
    ) -> None:
        self.policy_visualization.maybe_log_robot_state_xyz(
            device_name=device_name,
            state=state,
            timestamp_ns=timestamp_ns,
            summary_frame_timestamp_ns=summary_frame_timestamp_ns,
        )

    def _log_inference_visualizer_marker(self, **kwargs: Any) -> None:
        self.policy_visualization._log_inference_visualizer_marker(**kwargs)

    def _maybe_log_policy_reference_xyz(self) -> None:
        self.policy_visualization._maybe_log_policy_reference_xyz()

    def _maybe_log_policy_relative_xyz(self) -> None:
        self.policy_visualization._maybe_log_policy_relative_xyz()

    def _maybe_log_policy_history_relative_xyz(self, history_spec: Dict[str, Any]) -> None:
        self.policy_visualization._maybe_log_policy_history_relative_xyz(history_spec)

    def _is_robot_state_device(self, device_name: str) -> bool:
        return self.policy_visualization.is_robot_state_device(device_name)

    def _continuous_state_worker(self) -> None:
        """Continuously sample all robot states from summary SHM for visualization."""
        self.policy_visualization.continuous_state_worker()

    def _start_continuous_state_logging(self) -> None:
        self.policy_visualization.start_continuous_state_logging()

    def _stop_continuous_state_logging(self) -> None:
        self.policy_visualization.stop_continuous_state_logging()

    def _start_runtime_services(self) -> bool:
        return self.policy_loop._start_runtime_services()

    def _handle_runtime_control_signals(
        self,
        *,
        last_inference_state: bool,
        first_action_written: bool,
    ) -> Tuple[bool, bool, bool, bool]:
        return self.policy_loop._handle_runtime_control_signals(
            last_inference_state=last_inference_state,
            first_action_written=first_action_written,
        )

    def _prepare_next_policy_input(
        self,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int], bool]:
        return self.policy_loop._prepare_next_policy_input()

    def _predict_action_chunk(
        self,
        policy_input: Dict[str, Any],
        obs_time: int,
        iteration_count: int,
    ) -> Tuple[Dict[str, np.ndarray], int]:
        return self.policy_loop._predict_action_chunk(policy_input, obs_time, iteration_count)

    def _write_predicted_action_chunk(
        self,
        action_chunk_dict: Dict[str, np.ndarray],
        *,
        obs_time: int,
        inference_start_ns: int,
        iteration_count: int,
        first_action_written: bool,
    ) -> bool:
        return self.policy_loop._write_predicted_action_chunk(
            action_chunk_dict,
            obs_time=obs_time,
            inference_start_ns=inference_start_ns,
            iteration_count=iteration_count,
            first_action_written=first_action_written,
        )

    def _stop_runtime_services(self) -> None:
        self.policy_loop._stop_runtime_services()

    def run(self) -> None:
        self.policy_loop.run()
