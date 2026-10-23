#!/usr/bin/env python3
"""
Value prediction worker component for PolicyConnector.

Author: OpenAI Assistant
"""

from __future__ import annotations

import queue
from typing import Any

import numpy as np

from utils.logger_config import logger
from utils.time_control import precise_loop_timing


class ValuePredictionWorker:
    """Run background value prediction while keeping PolicyConnector state ownership."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def run(self) -> None:
        """Background thread worker for value prediction."""
        owner = self.owner
        wait_for_next_value_prediction = precise_loop_timing(owner.value_prediction_query_interval)
        while owner.value_prediction_running:
            try:
                obs_dict, obs_time = owner._assemble_observation_dict()
                if not obs_dict or obs_time is None:
                    logger.warning("PolicyConnector: Observation is None")
                    wait_for_next_value_prediction()
                    continue

                if not owner.inference_enabled:
                    wait_for_next_value_prediction()
                    continue

                policy_input = owner._prepare_policy_input(obs_dict, obs_time, history_with_value=False)

                action_dim = owner.action_configs[0]["action_dim"]
                predicted_action = np.zeros((owner.chunk_length, action_dim))

                if not owner.enable_value_prediction or not hasattr(owner.policy, "predict_value"):
                    logger.warning(
                        "PolicyConnector: Value prediction is disabled or policy does not have predict_value method"
                    )
                    continue

                try:
                    value_result = owner.policy_runtime.predict_value(policy_input, predicted_action)
                    value_data = value_result.get("value")
                    if value_data is None:
                        logger.warning("PolicyConnector: Value prediction result is None")
                        continue

                    self._write_value_result(value_data=value_data, obs_time=obs_time)
                except Exception as exc:
                    logger.warning(f"Value prediction failed: {exc}")

            except queue.Empty:
                continue
            except Exception as exc:
                logger.error(f"Error in value prediction worker: {exc}")
                continue

    def _write_value_result(self, *, value_data: Any, obs_time: int) -> None:
        """Write V-value or Q-values to the existing rollout SHM outputs."""
        owner = self.owner
        value_array = np.asarray(value_data, dtype=np.float64)

        is_q_function = len(value_array.shape) and value_array.shape[0] > 1
        if is_q_function:
            if value_array.ndim == 1:
                q_values = value_array.reshape(-1, 1)
            else:
                q_values = value_array
            action_horizon = q_values.shape[0]
            logger.info(f"Q-values reshaped to: {q_values.shape}")

            q_timestamps = np.array([
                obs_time + i * owner.action_step_interval_ns
                for i in range(action_horizon)
            ], dtype=np.int64)

            if (
                not owner.q_value_shm.created
                or (owner.q_value_shm.shape is not None and owner.q_value_shm.shape != q_values.shape)
            ):
                owner.q_value_shm.create(q_values.shape)
            if not owner.q_value_timestamps_shm.created:
                owner.q_value_timestamps_shm.create((action_horizon,))

            if owner.q_value_shm.created and owner.q_value_timestamps_shm.created:
                owner.q_value_shm.write_multi([q_values, q_timestamps], q_timestamps)
                logger.info(f"Q-values written to SHM: shape={q_values.shape}")
            else:
                logger.warning(
                    "Q-value SHM not created: "
                    f"q_value={owner.q_value_shm.created}, "
                    f"timestamps={owner.q_value_timestamps_shm.created}"
                )
            return

        if value_array.ndim == 0:
            value_array = value_array.reshape(1)
        if not owner.value_shm.created:
            owner.value_shm.create(value_array.shape)

        if owner.value_shm.created:
            owner.value_shm.write(value_array, obs_time)
        else:
            logger.warning(f"Value SHM not created: value={owner.value_shm.created}")
