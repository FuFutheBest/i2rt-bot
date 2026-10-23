#!/usr/bin/env python3
"""PolicyConnector inference-loop and lifecycle orchestration.

Author: Jun Lv, Han Xue, Zheng Wang
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from utils.logger_config import logger
from utils.time_control import precise_loop_timing


class PolicyLoop:
    """Own runtime lifecycle and one-iteration policy orchestration."""

    def __init__(self, owner: Any) -> None:
        object.__setattr__(self, "owner", owner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.owner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "owner":
            object.__setattr__(self, name, value)
            return
        setattr(self.owner, name, value)

    def _start_runtime_services(self) -> bool:
        """Start rollout, SHM connections, warm-up, and background services."""
        if self._should_start_infer_rollout():
            self._start_infer_rollout()

        if not self.connect():
            logger.error("PolicyConnector failed to connect to summary SHM")
            return False

        # FastWAM can optionally persist synchronized observations for offline
        # comparisons; the recorder is a no-op when disabled.
        recorder = getattr(self, "observation_recorder", None)
        if recorder is not None:
            try:
                self.observation_recorder_dir = recorder.start(
                    self._read_observation_recorder_frame
                )
                if self.observation_recorder_dir is not None:
                    recorder.mark_episode(
                        getattr(self, "_episode_count", 0),
                        getattr(self, "_history_episode_start_timestamp_ns", time.time_ns()),
                    )
            except Exception as exc:
                logger.warning(f"PolicyConnector: observation recorder failed to start: {exc}")

        if self.inference_enabled:
            logger.info("PolicyConnector: Inference is enabled; policy inference will begin now.")
        else:
            logger.info("PolicyConnector: Inference is disabled; waiting for inference to be enabled.")

        self._connect_policy_shm()
        self._start_continuous_state_logging()

        cached_policy_input: Optional[Dict[str, Any]] = None
        initial_obs_dict, initial_obs_time = self._assemble_observation_dict()
        if initial_obs_dict and initial_obs_time is not None:
            cached_policy_input = self._prepare_policy_input(
                initial_obs_dict,
                initial_obs_time,
                history_with_value=self.history_with_value,
            )
            if cached_policy_input is None:
                logger.warning("PolicyConnector: Failed to prepare initial policy input for warm-up")
        else:
            logger.warning("PolicyConnector: Failed to assemble initial observation for warm-up")

        enable_rtc = getattr(self.policy, 'enable_rtc', False)
        if enable_rtc and hasattr(self.policy, 'warm_up'):
            try:
                logger.info("PolicyConnector: Starting policy warm-up with RTC...")
                self.policy_runtime.warm_up(
                    self.action_configs,
                    num_observations=10,
                    sample_observation=cached_policy_input,
                )
                logger.info("PolicyConnector: Policy warm-up completed")
            except Exception as e:
                logger.warning(f"PolicyConnector: Warm-up failed, continuing anyway: {e}")

        logger.info(f"PolicyConnector running with {self.fps} Hz inference rate. Press Ctrl+C to stop...")
        logger.info(f"PolicyConnector: enable_value_prediction: {self.enable_value_prediction}")
        logger.info(f"PolicyConnector: policy has predict_value: {hasattr(self.policy, 'predict_value')}")

        if self.enable_value_prediction and hasattr(self.policy, 'predict_value'):
            self.value_prediction_running = True
            self.value_prediction_thread = threading.Thread(
                target=self._value_prediction_worker,
                daemon=True,
                name="ValuePredictionThread",
            )
            self.value_prediction_thread.start()
            logger.info("PolicyConnector: Started value prediction thread")

        if self.episode_control_enabled:
            self._setup_keyboard_listener()
            logger.info(f"PolicyConnector: Episode control enabled. Episode {self._episode_count} starting...")

        return True

    def _handle_runtime_control_signals(
        self,
        *,
        last_inference_state: bool,
        first_action_written: bool,
    ) -> Tuple[bool, bool, bool, bool]:
        """Handle reset/reconnect/episode/inference-state signals for one loop."""
        reset_iteration_count = False
        if self.episode_controller.check_reset_signal():
            logger.info("PolicyConnector: Reset signal detected, reconnecting all SHMs...")
            self.episode_controller.reconnect_all_shms()
            self._last_continuous_state_timestamps.clear()
            if hasattr(self.policy, 'reset'):
                self.policy.reset()
                logger.info("PolicyConnector: Reset policy's RTC state")
            self.policy_history_manager.clear_state()
            self.episode_controller.signal_policy_runner_reset()
            logger.info("PolicyConnector: Signaled policy runner to reset action cache")
            reset_iteration_count = True
            logger.info("PolicyConnector: Reset iteration_count to 0")
            # A panel-issued standalone reset deliberately pauses prediction
            # while the robot is moving Home.  The robot's reset signal is the
            # completion acknowledgement; resume only after it is consumed.
            if getattr(self, "_standalone_reset_pending", False):
                self._standalone_reset_pending = False
                self.inference_enabled = True
                command_id = getattr(self, "_standalone_reset_command_id", "")
                consumer = getattr(self, "inference_control_consumer", None)
                if command_id and consumer is not None:
                    try:
                        consumer.update_applied_state(
                            command_id,
                            inference_paused=False,
                            reset_pending=False,
                        )
                    except Exception as exc:
                        logger.warning(
                            "PolicyConnector: Failed to publish reset completion state: {}",
                            exc,
                        )
                self._standalone_reset_command_id = None
                logger.info(
                    "PolicyConnector: Standalone reset completed; inference resumed"
                )

        self.episode_controller.check_and_consume_robot_reconnect_signal()

        if self.episode_control_enabled and not self._episode_in_transition:
            episode_result = self.episode_controller.check_episode_end_request()
            if episode_result is not None:
                self.episode_controller.handle_episode_transition(episode_result)
                return False, last_inference_state, False, reset_iteration_count

        if self.inference_enabled != last_inference_state:
            if self.inference_enabled:
                logger.info("PolicyConnector: Inference ENABLED - model will perform predictions")
            else:
                logger.info("PolicyConnector: Inference DISABLED - model remains loaded but predictions are paused")
                if self._infer_rollout_started and first_action_written:
                    self._signal_stop_recording()
                    first_action_written = False
                self.policy_history_manager.clear_state()
            last_inference_state = self.inference_enabled

        return True, last_inference_state, first_action_written, reset_iteration_count

    def _prepare_next_policy_input(
        self,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int], bool]:
        """Assemble observation, update history if needed, and build policy input."""
        obs_dict, obs_time = self._assemble_observation_dict()
        if not obs_dict or obs_time is None:
            return None, None, True

        if not self.inference_enabled:
            return None, None, True

        block = self.policy.block if hasattr(self.policy, "block") else True
        if not self.policy_history_manager.maybe_update_policy_history(
            latest_obs_time=obs_time,
            block=block,
            current_obs=obs_dict,
        ):
            return None, None, False

        policy_input = self._prepare_policy_input(
            obs_dict,
            obs_time,
            self.history_with_value,
        )
        if policy_input is None:
            return None, None, True
        return policy_input, obs_time, False

    def _predict_action_chunk(
        self,
        policy_input: Dict[str, Any],
        obs_time: int,
        iteration_count: int,
    ) -> Tuple[Dict[str, np.ndarray], int]:
        """Run policy prediction with RTC action-index handling and logging."""
        action_index = self._get_prediction_action_index(iteration_count)

        if hasattr(self.policy, "set_prediction_context"):
            self.policy.set_prediction_context(timestamp_ns=obs_time)

        inference_start_ns = time.time_ns()
        self._log_inference_visualizer_marker(
            inference_event_type="inference_start",
            timestamp_ns=inference_start_ns,
            observation_timestamp_ns=int(obs_time),
            action_index=action_index,
        )
        try:
            action_chunk_dict = self.policy_runtime.predict(
                policy_input,
                action_configs=self.action_configs,
                chunk_length=self.chunk_length,
                action_index=action_index,
            )
        except Exception as exc:
            inference_end_ns = time.time_ns()
            self._log_inference_visualizer_marker(
                inference_event_type="inference_end",
                timestamp_ns=inference_end_ns,
                observation_timestamp_ns=int(obs_time),
                action_index=action_index,
                inference_start_timestamp_ns=inference_start_ns,
                success=False,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            raise

        profiler = getattr(self, "profiler", None)
        if profiler is not None:
            profiler.step(iteration_count)
        record_input = getattr(self, "_record_exact_policy_input", None)
        if callable(record_input):
            record_input(
                policy_input,
                int(obs_time),
                action_index=action_index,
                inference_start_timestamp_ns=inference_start_ns,
            )

        inference_end_ns = time.time_ns()
        timing_fields = self._prediction_timing_fields()
        timing_fields.update(
            self._record_inference_runtime_metrics(
                inference_start_ns,
                inference_end_ns,
            )
        )
        self._log_inference_visualizer_marker(
            inference_event_type="inference_end",
            timestamp_ns=inference_end_ns,
            observation_timestamp_ns=int(obs_time),
            action_index=action_index,
            inference_start_timestamp_ns=inference_start_ns,
            success=True,
            **timing_fields,
        )
        self._maybe_log_policy_relative_xyz()
        self._maybe_log_policy_reference_xyz()
        return action_chunk_dict, inference_start_ns

    def _record_inference_runtime_metrics(
        self,
        inference_start_ns: int,
        inference_end_ns: int,
    ) -> Dict[str, Any]:
        """Record wall-clock inference cadence and action-horizon coverage."""
        owner = self.owner
        inference_count = int(getattr(owner, "_inference_count", 0)) + 1
        owner._inference_count = inference_count
        if getattr(owner, "_first_inference_start_timestamp_ns", None) is None:
            owner._first_inference_start_timestamp_ns = int(inference_start_ns)
        owner._last_inference_start_timestamp_ns = int(inference_start_ns)

        actual_inference_hz: Optional[float] = None
        first_start_ns = getattr(owner, "_first_inference_start_timestamp_ns", None)
        if inference_count >= 2 and first_start_ns is not None:
            elapsed_ns = int(inference_start_ns) - int(first_start_ns)
            if elapsed_ns > 0:
                actual_inference_hz = float(inference_count - 1) * 1_000_000_000.0 / float(elapsed_ns)

        duration_ns = max(0, int(inference_end_ns) - int(inference_start_ns))
        control_fps = float(getattr(owner, "control_fps", 0.0) or 0.0)
        chunk_length = int(getattr(owner, "chunk_length", 0) or 0)
        horizon_ns: Optional[int] = None
        if control_fps > 0 and chunk_length > 0:
            horizon_ns = int(round(float(chunk_length) / control_fps * 1_000_000_000.0))
        exceeds_horizon = bool(horizon_ns is not None and duration_ns > horizon_ns)
        if exceeds_horizon:
            logger.warning(
                "PolicyConnector: inference duration exceeds action chunk horizon: "
                f"duration={duration_ns / 1e9:.3f}s, "
                f"horizon={horizon_ns / 1e9:.3f}s "
                f"(chunk_length={chunk_length}, control_fps={control_fps:.3f})"
            )

        return {
            "inference_count": inference_count,
            "configured_policy_fps": float(getattr(owner, "fps", 0.0) or 0.0),
            "actual_inference_hz": actual_inference_hz,
            "actual_policy_inference_hz": actual_inference_hz,
            "action_chunk_horizon_ns": horizon_ns,
            "action_chunk_horizon_s": None if horizon_ns is None else float(horizon_ns) / 1e9,
            "inference_exceeds_action_horizon": exceeds_horizon,
        }

    def _prediction_timing_fields(self) -> Dict[str, int]:
        """Expose optional policy-internal timings on the existing end marker."""
        raw_timing = getattr(self.policy, "last_prediction_timing_ns", {})
        if not isinstance(raw_timing, dict):
            return {}
        accepted = (
            "input_preprocess_ns",
            "h2d_ns",
            "cuda_forward_ns",
            "output_sync_d2h_ns",
            "postprocess_ns",
            "policy_predict_ns",
        )
        return {
            f"policy_{name}": int(raw_timing[name])
            for name in accepted
            if isinstance(raw_timing.get(name), (int, np.integer))
            and int(raw_timing[name]) >= 0
        }

    def _write_predicted_action_chunk(
        self,
        action_chunk_dict: Dict[str, np.ndarray],
        *,
        obs_time: int,
        inference_start_ns: int,
        iteration_count: int,
        first_action_written: bool,
    ) -> bool:
        """Write predicted action chunk and update related history/recording state."""
        if hasattr(self.policy, "get_history_update_spec"):
            self._pending_policy_history_spec = self.policy.get_history_update_spec(
                inference_start_timestamp_ns=inference_start_ns,
            )
        else:
            self._pending_policy_history_spec = None
        self._policy_history_arrival_wait_start_ns = None

        for device_name in action_chunk_dict:
            action_chunk_dict[device_name] = action_chunk_dict[device_name][:self.chunk_length, :]

        action_chunk_observation_timestamp = obs_time

        merged_chunk = self._merge_action_chunks(action_chunk_dict)
        if merged_chunk is not None and self.action_chunk_shm.created:
            self.action_chunk_shm.write(merged_chunk, obs_time)

        first_action_written = self._start_recording_on_first_action(first_action_written)

        action_chunk_sent_timestamp_ns = self._write_action_chunk_dict(
            action_chunk_dict,
            action_chunk_observation_timestamp,
        )
        if action_chunk_sent_timestamp_ns is not None:
            self.policy_history_manager.attach_action_chunk_timing(
                action_chunk_dict,
                first_action_sent_timestamp_ns=action_chunk_sent_timestamp_ns,
            )
        return first_action_written

    def _stop_runtime_services(self) -> None:
        """Stop threads, rollout, SHMs, and loggers used by run()."""
        self._stop_continuous_state_logging()
        if self.episode_control_enabled:
            self._stop_keyboard_listener()

        if self.value_prediction_running:
            self.value_prediction_running = False
            if self.value_prediction_thread is not None:
                try:
                    self.value_prediction_thread.join(timeout=2.0)
                    if self.value_prediction_thread.is_alive():
                        logger.warning("Value prediction thread did not stop gracefully")
                except Exception as e:
                    logger.warning(f"Error stopping value prediction thread: {e}")

        if self._infer_rollout_started:
            self._signal_stop_recording()
            time.sleep(0.5)
            self._stop_infer_rollout()
        recorder = getattr(self, "observation_recorder", None)
        if recorder is not None:
            try:
                recorder.stop()
            except Exception as exc:
                logger.warning(f"PolicyConnector: observation recorder failed to stop: {exc}")
        self.stop()
        if self.policy_shm:
            try:
                self.policy_shm.close()
            except Exception:
                pass
        self.value_shm.cleanup()
        self.action_chunk_shm.cleanup()
        self.q_value_shm.cleanup()
        self.q_value_timestamps_shm.cleanup()
        try:
            self.action_state_xyz_logger.close()
        except Exception as e:
            logger.warning(f"Error closing action/state xyz logger: {e}")
        for shm_name, control_shm in self._policy_history_control_shms.items():
            try:
                control_shm.close()
            except Exception as e:
                logger.warning(
                    f"Error closing policy history control SHM {shm_name}: {e}"
                )
        self._policy_history_control_shms.clear()
        close_policy = getattr(self.policy, "close", None)
        if callable(close_policy):
            try:
                close_policy()
            except Exception as exc:
                logger.warning(f"PolicyConnector: policy close failed: {exc}")
        profiler = getattr(self, "profiler", None)
        if profiler is not None:
            profiler.finalize()

    def run(self) -> None:
        if not self._start_runtime_services():
            return

        self.running = True
        wait_for_next_iteration = precise_loop_timing(self.update_interval)
        iteration_count = 0
        last_inference_state = self.inference_enabled
        first_action_written = False

        try:
            while self.running:
                self._process_inference_control_commands()
                should_continue, last_inference_state, first_action_written, reset_iteration_count = (
                    self._handle_runtime_control_signals(
                        last_inference_state=last_inference_state,
                        first_action_written=first_action_written,
                    )
                )
                if reset_iteration_count:
                    iteration_count = 0
                if not should_continue:
                    wait_for_next_iteration()
                    continue

                policy_input, obs_time, should_wait = self._prepare_next_policy_input()
                if policy_input is None or obs_time is None:
                    if should_wait:
                        wait_for_next_iteration()
                    continue

                action_chunk_dict, inference_start_ns = self._predict_action_chunk(
                    policy_input,
                    obs_time,
                    iteration_count,
                )
                first_action_written = self._write_predicted_action_chunk(
                    action_chunk_dict,
                    obs_time=obs_time,
                    inference_start_ns=inference_start_ns,
                    iteration_count=iteration_count,
                    first_action_written=first_action_written,
                )
                iteration_count += 1
                wait_for_next_iteration()

        except KeyboardInterrupt:
            logger.info("Stopping PolicyConnector...")
        finally:
            self._stop_runtime_services()
