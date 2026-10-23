"""OpenPi dual-arm UMI inference with image observation history.

This policy keeps the robot/control conversion from :class:`OpenPiUmiPolicy`,
but packs both wrist cameras as an oldest-to-newest history window matching
``UmiDualArmMemoryInputs`` in ``third_party/openpi_baseline``.  Only images carry
history; the robot state is the latest frame.
"""

from pathlib import Path
import time
from typing import Any, Dict, Optional

import numpy as np

from policy.openpi_umi_policy import (
    _RAW_LEFT,
    _RAW_RIGHT,
    OpenPiUmiPolicy,
    _as_uint8_hwc,
    _config,
    _crop_resize_wrist,
    logger,
    _maybe_dump_obs,
)


def _history_array(value: Any, *, key: str, horizon: int) -> np.ndarray:
    """Return an exact-length oldest-to-newest history array."""
    array = np.asarray(value)
    if array.ndim == 0 or array.shape[0] != horizon:
        length = array.shape[0] if array.ndim > 0 else 0
        raise ValueError(f"{key} history length {length}, expected {horizon}")
    return array


def _history_is_pad(
    observation: Dict[str, Any], *, key: str, horizon: int
) -> np.ndarray:
    pad_key = f"{key}_is_pad"
    if pad_key not in observation:
        return np.zeros(horizon, dtype=bool)
    pads = _history_array(observation[pad_key], key=pad_key, horizon=horizon)
    return pads.astype(bool, copy=False).reshape(horizon)


class OpenPiUmiHistoryPolicy(OpenPiUmiPolicy):
    """Dual-arm UMI policy for Pi0.5 checkpoints trained with image memory."""

    def __init__(
        self,
        *args: Any,
        expected_openpi_source: Optional[str] = None,
        debug_image_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.expected_openpi_source = expected_openpi_source
        self.debug_image_dir = (
            Path(debug_image_dir).expanduser() if debug_image_dir else None
        )
        self._debug_image_index = 0
        if self.debug_image_dir is not None:
            self.debug_image_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"OpenPi memory policy-input image dump enabled: {self.debug_image_dir}"
            )
        self._validate_openpi_source()
        super().__init__(*args, **kwargs)

        model_horizon = int(getattr(self.pi_config.model, "observation_horizon", 1))
        if model_horizon <= 1:
            raise ValueError(
                f"OpenPiUmiHistoryPolicy requires model observation_horizon > 1, got {model_horizon}"
            )
        if self.obs_horizon != model_horizon:
            raise ValueError(
                f"Runtime obs_horizon={self.obs_horizon} does not match model "
                f"observation_horizon={model_horizon}"
            )

    def _validate_openpi_source(self) -> None:
        if not self.expected_openpi_source:
            return
        expected = Path(self.expected_openpi_source).expanduser()
        if not expected.is_absolute():
            expected = Path(__file__).resolve().parents[1] / expected
        expected = expected.resolve()
        loaded = Path(_config.__file__).resolve()
        if expected not in loaded.parents:
            raise RuntimeError(
                "OpenPiUmiHistoryPolicy loaded openpi from the wrong source: "
                f"loaded={loaded}, expected under={expected}"
            )

    def _dump_debug_input_images(
        self, left_images: np.ndarray, right_images: np.ndarray
    ) -> None:
        """Save one left-right policy-input PNG for every history time step."""
        output_dir = getattr(self, "debug_image_dir", None)
        if output_dir is None:
            return
        if left_images.shape != right_images.shape:
            logger.warning(
                "Cannot dump OpenPi memory input: left/right history shapes differ: "
                f"left={left_images.shape}, right={right_images.shape}"
            )
            return
        if left_images.shape[0] != self.obs_horizon:
            logger.warning(
                "Cannot dump OpenPi memory input: unexpected history length: "
                f"got={left_images.shape[0]}, expected={self.obs_horizon}"
            )
            return

        import cv2

        inference_index = self._debug_image_index
        timestamp_ns = time.time_ns()
        for history_index in range(self.obs_horizon):
            left = _as_uint8_hwc(left_images[history_index])
            right = _as_uint8_hwc(right_images[history_index])
            combined_rgb = np.concatenate([left, right], axis=1)
            combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
            position = "latest" if history_index == self.obs_horizon - 1 else "history"
            output_path = output_dir / (
                f"policy_input_{inference_index:06d}_{timestamp_ns}_"
                f"t{history_index:02d}_{position}_left_right.png"
            )
            if not cv2.imwrite(str(output_path), combined_bgr):
                logger.warning(f"Failed to write OpenPi memory debug image: {output_path}")
            else:
                logger.info(
                    f"Wrote OpenPi memory debug image: {output_path} "
                    f"shape={combined_rgb.shape}"
                )
        self._debug_image_index += 1

    def _convert_obs_to_pi(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        # PolicyConnector preserves config obs.devices order and returns history
        # oldest -> newest.  Ignore the generated *_is_pad entries when selecting
        # the two physical camera streams.
        cam_keys = [
            key
            for key in observation
            if "camera" in key.lower() and not key.endswith("_is_pad")
        ]
        if len(cam_keys) < 2:
            raise ValueError(f"need 2 wrist cameras, got: {cam_keys}")

        processed_images: Dict[str, np.ndarray] = {}
        image_histories = []
        for side, camera_key in zip(("left", "right"), cam_keys[:2], strict=True):
            raw_history = _history_array(
                observation[camera_key], key=camera_key, horizon=self.obs_horizon
            )
            frames = []
            for index, raw_frame in enumerate(raw_history):
                device_frame = _as_uint8_hwc(raw_frame)
                model_frame = _crop_resize_wrist(device_frame)
                frames.append(model_frame)
                processed_images[f"{side}_device_t{index}"] = device_frame
                processed_images[f"{side}_img_t{index}"] = model_frame
            image_histories.append(np.stack(frames, axis=0))

        _maybe_dump_obs(processed_images)
        left_images, right_images = image_histories
        self._dump_debug_input_images(left_images, right_images)

        robot_key = next(
            (
                key
                for key in observation
                if key.lower().startswith(("rizon", "robot"))
                and not key.endswith("_is_pad")
            ),
            None,
        )
        if robot_key is None:
            raise KeyError("Robot observation data not found in observation dictionary")
        robot_history = _history_array(
            observation[robot_key], key=robot_key, horizon=self.obs_horizon
        )
        robot_obs = np.asarray(robot_history[-1])
        if robot_obs.ndim != 1:
            raise ValueError(
                f"Latest robot observation must be 1-D, got shape {robot_obs.shape}"
            )

        lpos, lquat, lgrip = self._arm_to_umi(robot_obs[_RAW_LEFT])
        rpos, rquat, rgrip = self._arm_to_umi(robot_obs[_RAW_RIGHT])
        ego = np.zeros(7, dtype=np.float64)
        state23 = np.concatenate(
            [lpos, lquat, lgrip, rpos, rquat, rgrip, ego]
        ).astype(np.float32)
        if self.mask_state:
            state23 = np.zeros_like(state23)

        return {
            "state": state23,
            "left_wrist_image": left_images,
            "right_wrist_image": right_images,
            "left_wrist_image_is_pad": _history_is_pad(
                observation, key=cam_keys[0], horizon=self.obs_horizon
            ),
            "right_wrist_image_is_pad": _history_is_pad(
                observation, key=cam_keys[1], horizon=self.obs_horizon
            ),
            "prompt": self.task,
        }
