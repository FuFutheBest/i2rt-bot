#!/usr/bin/env python3
"""
Observation builder for converting summary SHM reads into policy observations.

Author: OpenAI Assistant
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from utils.logger_config import logger


ReadAllDeviceData = Callable[..., Optional[Tuple[int, Dict[str, Tuple[int, np.ndarray]]]]]
RobotStateLogger = Callable[..., None]


class ObservationBuilder:
    """Build the latest observation dict from BaseConsumer summary reads."""

    def __init__(
        self,
        *,
        read_all_device_data: ReadAllDeviceData,
        master_device: Optional[str],
        robot_state_logger: Optional[RobotStateLogger] = None,
        owner: Optional[Any] = None,
    ) -> None:
        self.read_all_device_data = read_all_device_data
        self.master_device = master_device
        self.robot_state_logger = robot_state_logger
        self.owner = owner
        self._history_reset_pending = False
        self._history_start_timestamp_ns: Optional[int] = None

    def __getattr__(self, name: str) -> Any:
        if self.owner is None:
            raise AttributeError(name)
        return getattr(self.owner, name)

    def assemble(self) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[int]]:
        """Assemble a policy observation dict and its observation timestamp."""
        try:
            read_result = self.read_all_device_data(include_frame_timestamp=True)
            if read_result is None:
                return None, None
            frame_timestamp_ns, all_data = read_result
            if all_data is None:
                return None, None

            obs_dict: Dict[str, np.ndarray] = {}
            all_obs_time = []
            for device_name, (timestamp, data) in all_data.items():
                obs_dict[device_name] = data
                if self.master_device is None:
                    all_obs_time.append(timestamp)
                if self.robot_state_logger is not None:
                    self.robot_state_logger(
                        device_name=device_name,
                        state=data,
                        timestamp_ns=int(timestamp),
                        summary_frame_timestamp_ns=int(frame_timestamp_ns),
                    )

            if self.master_device is None:
                obs_time = int(np.array(all_obs_time).mean().item())
            else:
                obs_time = int(all_data[self.master_device][0])
            record_metadata = getattr(self.owner, "_record_observation_metadata", None)
            if callable(record_metadata):
                record_metadata(
                    int(obs_time),
                    int(frame_timestamp_ns),
                    all_data,
                )
            return obs_dict, obs_time

        except Exception as exc:
            logger.error(f"Error assembling observation dict: {exc}")
            return None, None

    def _requires_history(self) -> bool:
        obs_horizon = getattr(self.policy, 'obs_horizon', 1)
        return (
            isinstance(obs_horizon, int) and obs_horizon > 1 and
            hasattr(self.policy, 'obs_time_interval') and
            (
                hasattr(self.policy, 'condition_history_stride')
                or hasattr(self.policy, 'downsample')
            )
        )

    def reset_history(self) -> None:
        """Prevent summary frames from a completed episode entering the next one."""
        self._history_reset_pending = True
        self._history_start_timestamp_ns = None

    def _collect_observation_history(
        self,
        latest_obs: Dict[str, np.ndarray],
        latest_timestamp: int,
        history_with_value: bool = False,
    ) -> Optional[Dict[str, List[np.ndarray]]]:
        with self.summary_shm_lock:
            return self._collect_observation_history_unlocked(
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
        '''
        Collect observation history from SHM while holding the summary SHM lock.
        '''

        obs_horizon = getattr(self.policy, 'obs_horizon', 1)
        if not isinstance(obs_horizon, int) or obs_horizon <= 1:
            return {k: [np.array(v, copy=True)] for k, v in latest_obs.items()}

        interval_sec = float(getattr(self.policy, 'obs_time_interval', 0.0))
        condition_history_stride = int(
            getattr(
                self.policy,
                'condition_history_stride',
                getattr(self.policy, 'downsample', 1),
            )
        )
        if interval_sec <= 0.0 or condition_history_stride <= 0:
            logger.warning("PolicyConnector: Invalid observation interval configuration; falling back to single observation")
            return {k: [np.array(v, copy=True)] for k, v in latest_obs.items()}
        dt_ns = int(interval_sec * condition_history_stride * 1e9)

        if self._history_reset_pending:
            self._history_start_timestamp_ns = int(latest_timestamp)
            self._history_reset_pending = False

        value_prediction_key = "value_prediction" if history_with_value else None
        latest_obs = {key: np.array(value, copy=True) for key, value in latest_obs.items()}
        obs_list: List[Dict[str, np.ndarray]] = [latest_obs]
        history_frame_timestamps_ns = [int(latest_timestamp)]

        # obs_list[0] should have an 'pad' value prediction, we implement it in the end of the function, by modifying the obs_batch
        if history_with_value:
            obs_list[0][value_prediction_key] = np.zeros(1, dtype=np.float32)
        is_pad_list: Optional[List[bool]] = [False]
        history_obs_times = [latest_timestamp - (i + 1) * dt_ns for i in range(obs_horizon - 1)]

        # collect history observations
        for target_timestamp in history_obs_times:
            past_frame_invalid = False
            past_value_invalid = False
            past_obs = None
            past_value = None
            if (
                self._history_start_timestamp_ns is not None
                and target_timestamp < self._history_start_timestamp_ns
            ):
                past_frame = None
            else:
                past_frame = self.find_frame_by_timestamp(
                    target_timestamp,
                    tolerance_ms=50.0,
                )
            if past_frame is None:
                logger.warning("PolicyConnector: the requested history frame is not found, too old, or the shm is empty, we pad it")
                past_frame_invalid = True
            # the action_policy requires history frame with the corresponding value prediction
            if history_with_value:
                past_value = self.find_value_by_timestamp_with_history_too_old_detection(target_timestamp, tolerance_ms=50.0)
                if past_value is None:
                    logger.warning("PolicyConnector: past_value is not found, too old, or the shm is empty, we pad it")
                    past_value_invalid = True
            # if the history is too old, we need to contruct a 'pad' frame for the policy input, marking the corresponding observation as 'pad'
            if past_frame_invalid or past_value_invalid:
                # take out value from the last obs in the obs_list
                past_obs = {key: np.array(value, copy=True) for key, value in obs_list[-1].items()}
                if history_with_value:
                    # add the 'pad' value prediction
                    past_obs[value_prediction_key] = np.zeros(1, dtype=np.float32)
                # add data and the pad flag into the list
                obs_list.append(past_obs)
                is_pad_list.append(True)
                # A padded observation is a copy of the newest available
                # source frame. Keep its real source timestamp; ``*_is_pad``
                # carries the intended temporal-position information.
                history_frame_timestamps_ns.append(
                    history_frame_timestamps_ns[-1]
                )
            else:
                # past_frame['data'] is a tuple, with idx 0 the frame timestamp, and idx 1 the frame array
                # take out the value from (the past frame and the past value data)
                past_obs = {key: np.array(value[1], copy=True) for key, value in past_frame['data'].items()}
                if history_with_value:
                    # add the 'pad' value prediction
                    assert isinstance(past_value, np.ndarray), f"past_value must be a numpy array, got {type(past_value)}"
                    past_obs[value_prediction_key] = np.array(past_value, copy=True)
                # add data and the pad flag into the list
                obs_list.append(past_obs)
                # logger.info(f"PolicyConnector: collected history observation: {past_obs}")
                is_pad_list.append(False)
                history_frame_timestamps_ns.append(
                    int(past_frame.get('frame_timestamp_ns', target_timestamp))
                )

        # Post data processing, process the obs_list and the pad flag list into the obs_batch
        if len(obs_list) != obs_horizon:
            logger.warning(f"PolicyConnector: Requested history is not collected; obs_list: {len(obs_list)}, obs_horizon: {obs_horizon}")
            return None
        if is_pad_list is None or len(is_pad_list) != obs_horizon:
            logger.warning(
                f"PolicyConnector: Requested stride history, but is_pad_list: {len(is_pad_list) if is_pad_list is not None else None}, obs_horizon: {obs_horizon}"
            )
            return None

        # reverse the obs_list and the pad list, let the latest observation be the [-1] element
        obs_list.reverse()
        history_frame_timestamps_ns.reverse()
        if is_pad_list is not None:
            is_pad_list.reverse()
        obs_batch: Dict[str, List[np.ndarray]] = {}
        for key in latest_obs.keys():
            obs_batch[key] = [step[key] for step in obs_list]
            obs_batch[f"{key}_is_pad"] = [np.asarray(is_pad, dtype=bool) for is_pad in is_pad_list]
        obs_batch["_history_frame_timestamps_ns"] = np.asarray(
            history_frame_timestamps_ns,
            dtype=np.int64,
        )
        # add the value prediction and the value prediction pad flag into the obs_batch
        if history_with_value:
            obs_batch[value_prediction_key] = [step[value_prediction_key] for step in obs_list]
            obs_batch[f"{value_prediction_key}_is_pad"] = [np.asarray(is_pad, dtype=bool) for is_pad in is_pad_list]
            # # Here we need to set the latest value prediction as the 'pad' value prediction, to be coherent with the training process, but we have surpassed it because it is now handled in data transform from open pi
            # obs_batch[f"{value_prediction_key}_is_pad"][-1] = True

        return obs_batch

    def _prepare_policy_input(
        self,
        latest_obs: Dict[str, np.ndarray],
        latest_timestamp: int,
        history_with_value: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not latest_obs:
            return None

        if self._requires_history():
            history = self._collect_observation_history(latest_obs, latest_timestamp, history_with_value=history_with_value)
            # logger.info(f"collected obs history: {history}")
            if history is None:
                return None
            return history

        return latest_obs
