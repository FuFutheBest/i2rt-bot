#!/usr/bin/env python3
"""Rizon robot helper mixin.

Author: OpenAI Assistant
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from utils.logger_config import logger
from utils.lk_gripper_presets import get_lk_gripper_preset

Gripper = None
LKGripperConfig = None
LK_GRIPPER_IMPORT_ERROR = None
REPO_ROOT = Path(__file__).resolve().parents[3]


class RizonGripperMixin:
    """Gripper calibration and command helpers for RizonRobot."""

    def _load_lk_gripper_calibration_file(self, index: int) -> Dict[str, Any]:
        """Load LK gripper calibration JSON for an arm index."""
        path_value = self.gripper_config.arms[index].calibration_file
        if path_value is None or str(path_value).strip() == "":
            return {}

        calibration_path = Path(str(path_value)).expanduser()
        if not calibration_path.is_absolute():
            calibration_path = REPO_ROOT / calibration_path
        if not calibration_path.is_file():
            raise FileNotFoundError(f"LK gripper calibration file not found: {calibration_path}")

        with calibration_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        logger.info(f"Loaded LK gripper calibration[{index}] from {calibration_path}")
        return data

    @staticmethod
    def _calibration_value(data: Dict[str, Any], key: str, fallback: Any = None) -> Any:
        return data[key] if key in data else fallback

    def _apply_lk_gripper_calibration_files(self) -> None:
        """Apply LK calibration JSON metadata to runtime conversion parameters."""
        if self.gripper_config.backend != "lktech":
            return

        self._lk_gripper_calibrations = []
        for index, arm_config in enumerate(self.gripper_config.arms):
            calibration = self._load_lk_gripper_calibration_file(index)
            self._lk_gripper_calibrations.append(calibration)
            if not calibration:
                continue

            if calibration.get("gripper_type"):
                self.gripper_config.gripper_type = str(
                    calibration["gripper_type"]
                ).strip().lower()
            for key in (
                "min_torque_iq",
                "max_torque_iq",
                "min_encoder",
                "max_encoder",
                "min_angle_deg",
                "max_angle_deg",
                "arm_length_m",
                "apex_angle_max_deg",
                "gear_ratio",
            ):
                if key in calibration:
                    setattr(arm_config, key, calibration[key])

    @staticmethod
    def _get_lk_gripper_classes():
        """Import LK gripper classes lazily so non-LK configs do not require pyserial."""
        global Gripper, LKGripperConfig, LK_GRIPPER_IMPORT_ERROR
        if Gripper is not None and LKGripperConfig is not None:
            return LKGripperConfig, Gripper
        try:
            from third_party.lktech_motor_sdk.example.lktech_gripper import (
                Gripper as ImportedGripper,
                LKGripperConfig as ImportedLKGripperConfig,
            )
        except ImportError as exc:
            LK_GRIPPER_IMPORT_ERROR = exc
            return None, None
        Gripper = ImportedGripper
        LKGripperConfig = ImportedLKGripperConfig
        LK_GRIPPER_IMPORT_ERROR = None
        return LKGripperConfig, Gripper

    def _get_lk_gripper_arm_length(self, index: int) -> float:
        """Return cosine-law arm length for the LK SDK width model."""
        configured = self.gripper_config.arms[index].arm_length_m
        if configured is not None:
            return float(configured)
        preset = get_lk_gripper_preset(self.gripper_config.gripper_type)
        if preset.get("arm_length_m") is None:
            raise ValueError(f"gripper arm_length_m is required for index {index}")
        return float(preset["arm_length_m"])

    def _switch_tool_if_needed(self, robot, tool, tool_name: str, label: str) -> None:
        """Switch the robot tool if a tool name is configured."""
        if tool is None or not tool_name:
            return

        if robot is not None:
            try:
                logger.info(f"Switching {label} mode to [IDLE] before tool switch")
                robot.switch_mode(self.flexiv_adapter.mode_idle)
                time.sleep(0.2)
            except Exception as exc:
                logger.warning(f"Failed to switch {label} to IDLE before tool switch: {exc}")

        logger.info(f"Switching {label} tool to [{tool_name}]")
        tool.switch(tool_name)

    def _move_flexiv_gripper_home(self, gripper, index: int, label: str) -> bool:
        """Move a Flexiv SDK gripper to the configured physical home width."""
        if gripper is None:
            return False
        arm_config = self.gripper_config.arms[index]
        home_width = self._get_home_gripper_width(index)
        logger.info(f"Resetting {label} flexiv gripper to home width ({home_width:.4f} m)")
        gripper.move(home_width, arm_config.max_vel, arm_config.force_limit)
        self.last_gripper_value[index] = self._target_to_measured_gripper_width(home_width)
        self.last_gripper_command_width[index] = home_width
        return True

    def _read_flexiv_gripper_value(self, gripper, index: int, label: str) -> float:
        """Read a Flexiv SDK gripper using legacy width semantics."""
        if gripper is None:
            return float(self.last_gripper_value[index])
        try:
            raw_width = gripper.states().width
            gripper_value = self._target_to_measured_gripper_width(raw_width)
            self.last_gripper_value[index] = gripper_value
            return gripper_value
        except Exception as e:
            logger.warning(f"Failed to read {label} flexiv gripper value: {e}")
            return float(self.last_gripper_value[index])

    def _move_gripper_home_by_backend(self, gripper, index: int, label: str, wait: bool = False) -> bool:
        """Move a gripper home using the active backend."""
        if self.gripper_config.backend == "lktech":
            return self._move_gripper_home(gripper, index, label, wait=wait)
        return self._move_flexiv_gripper_home(gripper, index, label)

    def _read_gripper_value_by_backend(self, gripper, index: int, label: str) -> float:
        """Read a gripper value using the active backend."""
        if self.gripper_config.backend == "lktech":
            return self._read_gripper_value(gripper, index, label)
        return self._read_flexiv_gripper_value(gripper, index, label)

    def _move_gripper_value_by_backend(self, gripper, gripper_value: float, index: int, label: str) -> bool:
        """Command a gripper value using the active backend."""
        if self.gripper_config.backend == "lktech":
            return self._move_gripper_value(gripper, gripper_value, index, label)

        if gripper is None:
            logger.warning(f"{label} gripper is not available")
            return False
        try:
            arm_config = self.gripper_config.arms[index]
            target_width = self._measured_to_target_gripper_width(float(gripper_value), index=index)
            gripper.move(
                target_width,
                arm_config.max_vel,
                arm_config.force_limit,
            )
            self.last_gripper_value[index] = self._target_to_measured_gripper_width(target_width)
            self.last_gripper_command_width[index] = target_width
            return True
        except Exception as e:
            logger.error(f"Failed to move {label} flexiv gripper to value {gripper_value}: {e}")
            return False

    def _width_to_normalized_gripper_position(self, width: float, index: int) -> float:
        """Convert a physical gripper width in meters to repo/LK-facing normalized position."""
        arm_config = self.gripper_config.arms[index]
        min_width = float(arm_config.min_width)
        max_width = float(arm_config.max_width)
        if max_width <= min_width:
            raise ValueError(
                f"Invalid gripper width range for index {index}: min={min_width}, max={max_width}"
            )

        clipped_width = float(np.clip(width, min_width, max_width))
        open_ratio = (clipped_width - min_width) / (max_width - min_width)
        return float(np.clip(open_ratio, 0.0, 1.0))

    def _normalized_gripper_position_to_width(self, normalized_position: float, index: int) -> float:
        """Convert normalized position back to a physical gripper width in meters."""
        arm_config = self.gripper_config.arms[index]
        min_width = float(arm_config.min_width)
        max_width = float(arm_config.max_width)
        if max_width <= min_width:
            raise ValueError(
                f"Invalid gripper width range for index {index}: min={min_width}, max={max_width}"
            )

        clipped_position = float(np.clip(normalized_position, 0.0, 1.0))
        return float(min_width + clipped_position * (max_width - min_width))

    @staticmethod
    def _clip_normalized_gripper_position(normalized_position: float) -> float:
        return float(np.clip(normalized_position, 0.0, 1.0))

    def _get_home_gripper_width(self, index: int) -> float:
        """Return the physical home width in meters."""
        arm_config = self.gripper_config.arms[index]
        if self.gripper_config.value_mode == "width_m":
            return arm_config.home_width

        home_ratio = arm_config.home_ratio
        if home_ratio is None:
            return arm_config.home_width

        min_width = float(arm_config.min_width)
        max_width = float(arm_config.max_width)
        if max_width <= min_width:
            raise ValueError(
                f"Invalid gripper width range for index {index}: min={min_width}, max={max_width}"
            )

        clipped_ratio = float(np.clip(float(home_ratio), 0.0, 1.0))
        return float(min_width + clipped_ratio * (max_width - min_width))

    def _command_gripper_value_to_normalized_position(
        self, gripper_value: float, index: int
    ) -> float:
        """Convert the configured command unit into a normalized target."""
        if self.gripper_config.command_value_mode == "normalized":
            return self._clip_normalized_gripper_position(gripper_value)

        return self._width_to_normalized_gripper_position(gripper_value, index)

    def _normalized_position_to_feedback_gripper_value(
        self, normalized_position: float, index: int
    ) -> float:
        """Convert normalized state into the configured feedback unit."""
        if self.gripper_config.value_mode == "normalized":
            clipped_position = self._clip_normalized_gripper_position(normalized_position)
            return clipped_position

        return self._normalized_gripper_position_to_width(normalized_position, index)

    def _get_home_gripper_feedback_value(self, index: int) -> float:
        """Return the home value in the configured state-feedback unit."""
        home_width = self._get_home_gripper_width(index)
        if self.gripper_config.value_mode == "normalized":
            return self._width_to_normalized_gripper_position(home_width, index)
        return home_width

    def _get_home_gripper_command_value(self, index: int) -> float:
        """Return the home value in the configured command unit."""
        home_width = self._get_home_gripper_width(index)
        if self.gripper_config.command_value_mode == "normalized":
            return self._width_to_normalized_gripper_position(home_width, index)
        return home_width

    def _gripper_width_and_motor_angle(
        self, gripper, gripper_value: float, index: int
    ) -> tuple[float, float]:
        """Return physical opening width in meters and calibrated motor radians."""
        if self.gripper_config.value_mode == "width_m":
            width_m = float(gripper_value)
            try:
                normalized = float(gripper._normalized_from_width(width_m))
            except Exception:
                normalized = self._width_to_normalized_gripper_position(
                    width_m, index
                )
        else:
            normalized = self._clip_normalized_gripper_position(gripper_value)
            width_m = self._normalized_gripper_position_to_width(normalized, index)

        motor_radians = 0.0
        if gripper is not None:
            try:
                min_angle, max_angle, mode = gripper._get_calibrated_range()
                if str(mode) == "multi_turn_angle":
                    motor_degrees = float(min_angle) + (1.0 - normalized) * (
                        float(max_angle) - float(min_angle)
                    )
                    motor_radians = float(np.radians(motor_degrees))
            except Exception:
                pass
        return width_m, motor_radians

    def _scale_realtime_gripper_command(self, gripper_value: float, index: int) -> float:
        """Scale real-time policy gripper commands without affecting reset/calibration flows."""
        # NOTE: hardcode here
        arm_config = self.gripper_config.arms[index]
        if float(gripper_value) < 0.8:
            scaled_value = float(gripper_value) * arm_config.scale_factor
        else:
            scaled_value = float(gripper_value)
        # TODO(remove hack): temporary gripper close bias for 2026-06-26 inference tests.
        scaled_value = max(0.0, scaled_value - arm_config.command_offset)
        if self.gripper_config.command_value_mode == "normalized":
            return self._clip_normalized_gripper_position(scaled_value)

        return float(scaled_value)

    def _build_gripper_runtime_config(self, index: int, label: str):
        """Build runtime config for the bundled LK gripper wrapper."""
        arm_config = self.gripper_config.arms[index]
        gripper_config_cls, _ = self._get_lk_gripper_classes()
        if gripper_config_cls is None:
            raise ImportError(
                "LKTech gripper support requires the bundled third_party/lktech_motor_sdk "
                "and its Python dependencies, including pyserial."
            ) from LK_GRIPPER_IMPORT_ERROR
        calibration = (
            self._lk_gripper_calibrations[index]
            if hasattr(self, "_lk_gripper_calibrations") and index < len(self._lk_gripper_calibrations)
            else self._load_lk_gripper_calibration_file(index)
        )
        if calibration:
            cal_type = calibration.get("gripper_type")
            if (
                cal_type
                and str(cal_type).strip().lower()
                != self.gripper_config.gripper_type
            ):
                logger.warning(
                    f"LK gripper calibration[{index}] type {cal_type!r} differs from "
                    "configured gripper_type "
                    f"{self.gripper_config.gripper_type!r}; using calibration file values"
                )
        home_width = self._get_home_gripper_width(index)
        return gripper_config_cls(
            port=str(arm_config.port),
            motor_id=arm_config.motor_id,
            baudrate=arm_config.baudrate,
            speed_dps=arm_config.speed_dps,
            home_position=(
                self._width_to_normalized_gripper_position(home_width, index)
                if self.gripper_config.value_mode == "normalized"
                else 1.0
            ),
            read_timeout=arm_config.timeout,
            wait_timeout=arm_config.wait_timeout,
            auto_calibrate=arm_config.auto_calibrate,
            min_torque_iq=int(
                self._calibration_value(
                    calibration, "min_torque_iq", arm_config.min_torque_iq
                )
            ),
            max_torque_iq=int(
                self._calibration_value(
                    calibration, "max_torque_iq", arm_config.max_torque_iq
                )
            ),
            calibration_timeout=arm_config.calibration_timeout,
            calibration_settle_time=arm_config.calibration_settle_time,
            calibration_dir=arm_config.calibration_directory,
            min_encoder=(
                int(self._calibration_value(calibration, "min_encoder", arm_config.min_encoder))
                if self._calibration_value(calibration, "min_encoder", arm_config.min_encoder) is not None
                else None
            ),
            max_encoder=(
                int(self._calibration_value(calibration, "max_encoder", arm_config.max_encoder))
                if self._calibration_value(calibration, "max_encoder", arm_config.max_encoder) is not None
                else None
            ),
            min_angle_deg=(
                float(self._calibration_value(calibration, "min_angle_deg", arm_config.min_angle_deg))
                if self._calibration_value(calibration, "min_angle_deg", arm_config.min_angle_deg) is not None
                else None
            ),
            max_angle_deg=(
                float(self._calibration_value(calibration, "max_angle_deg", arm_config.max_angle_deg))
                if self._calibration_value(calibration, "max_angle_deg", arm_config.max_angle_deg) is not None
                else None
            ),
            arm_length_m=float(self._calibration_value(calibration, "arm_length_m", self._get_lk_gripper_arm_length(index))),
            apex_angle_max_deg=float(self._calibration_value(calibration, "apex_angle_max_deg", arm_config.apex_angle_max_deg)),
            gear_ratio=float(self._calibration_value(calibration, "gear_ratio", arm_config.gear_ratio)),
            label=label,
        )

    def _create_lk_gripper(self, robot, index: int, label: str):
        """Build an LK gripper instance from normalized config."""
        _, gripper_cls = self._get_lk_gripper_classes()
        if gripper_cls is None:
            raise ImportError(
                "LKTech gripper support requires the bundled third_party/lktech_motor_sdk "
                "and its Python dependencies, including pyserial."
            ) from LK_GRIPPER_IMPORT_ERROR
        config = self._build_gripper_runtime_config(index, label)
        arm_config = self.gripper_config.arms[index]
        return gripper_cls(
            robot,
            config=config,
            min_vel=arm_config.min_vel,
            max_vel=arm_config.max_vel,
        )

    def _move_gripper_value(self, gripper, gripper_value: float, index: int, label: str, wait: bool = False) -> bool:
        """Command the LK gripper using repo-facing gripper value semantics."""
        if gripper is None:
            logger.warning(f"{label} gripper is not available")
            return False

        try:
            arm_config = self.gripper_config.arms[index]
            if self.gripper_config.command_value_mode == "normalized":
                target_position = self._command_gripper_value_to_normalized_position(
                    float(gripper_value),
                    index,
                )
                target_width = self._normalized_gripper_position_to_width(target_position, index)
            else:
                target_width = self._measured_to_target_gripper_width(float(gripper_value), index=index)
                min_width = (
                    0.0
                    if arm_config.min_width is None
                    else float(arm_config.min_width)
                )
                max_width = arm_config.max_width
                if max_width is not None:
                    target_width = float(np.clip(target_width, min_width, float(max_width)))
                else:
                    target_width = max(min_width, float(target_width))
            measured_width = self._target_to_measured_gripper_width(target_width)
            if self.gripper_config.value_mode == "normalized":
                self.last_gripper_value[index] = self._width_to_normalized_gripper_position(
                    measured_width,
                    index,
                )
            else:
                self.last_gripper_value[index] = measured_width
            velocity = arm_config.max_vel
            force_limit = arm_config.force_limit
            ok = gripper.Move(target_width, velocity, force_limit)
            if ok:
                self.last_gripper_command_width[index] = target_width
            if wait and hasattr(gripper, "motor"):
                return bool(gripper.motor.wait_for_stop(timeout=gripper.wait_timeout))
            return bool(ok)
        except Exception as e:
            logger.error(f"Failed to move {label} gripper to value {gripper_value}: {e}")
            return False

    def _read_gripper_value(self, gripper, index: int, label: str) -> float:
        """Read the current repo-facing gripper value."""
        if gripper is None:
            return float(self.last_gripper_value[index])

        try:
            raw_width = float(gripper.states().width)
            if self.gripper_config.value_mode == "normalized":
                raw_position = self._width_to_normalized_gripper_position(raw_width, index)
                gripper_value = self._normalized_position_to_feedback_gripper_value(
                    raw_position, index
                )
            else:
                gripper_value = self._target_to_measured_gripper_width(raw_width)
            self.last_gripper_value[index] = gripper_value
            return gripper_value
        except Exception as e:
            logger.warning(f"Failed to read {label} gripper value: {e}")
            return float(self.last_gripper_value[index])

    def _move_gripper_home(self, gripper, index: int, label: str, wait: bool = False) -> bool:
        """Move a gripper to its configured home target."""
        home_value = self._get_home_gripper_command_value(index)
        home_width = self._get_home_gripper_width(index)
        ratio = self.gripper_config.arms[index].home_ratio
        ratio_text = "None" if ratio is None else f"{float(ratio):.4f}"
        logger.info(
            f"Resetting {label} gripper to home value "
            f"({home_value:.4f}, mode={self.gripper_config.value_mode}, "
            f"width_m={home_width:.4f}, ratio={ratio_text})"
        )
        return self._move_gripper_value(gripper, home_value, index, label, wait=wait)

    def _shutdown_grippers(self) -> None:
        """Power off and disconnect any initialized grippers."""
        gripper_attrs = ("gripper", "gripper_left", "gripper_right")
        for gripper_attr in gripper_attrs:
            gripper = getattr(self, gripper_attr, None)
            if gripper is None:
                continue

            disable_fn = getattr(gripper, "Disable", None)
            if disable_fn is None:
                disable_fn = getattr(gripper, "disable", None)
            if not callable(disable_fn):
                logger.warning(f"Gripper {gripper_attr} has no disable method")
                continue

            try:
                disable_fn()
            except Exception as e:
                logger.warning(f"Failed during {gripper_attr} shutdown: {e}")
            finally:
                setattr(self, gripper_attr, None)

    def _load_gripper_calibration_data(self) -> None:
        """Load gripper calibration data from JSON file."""
        import json
        try:
            calibration_path = self.gripper_config.inference_calibration_path
            with open(calibration_path, 'r') as f:
                calibration_json = json.load(f)

            data_points = calibration_json.get("data_points", [])
            if not data_points:
                logger.warning("No calibration data points found in file")
                return

            # Sort data points by target_width for interpolation
            sorted_points = sorted(data_points, key=lambda x: x["target_width"])

            self.gripper_calibration_data = {
                "target_widths": np.array([p["target_width"] for p in sorted_points]),
                "measured_widths": np.array([p["measured_width"] for p in sorted_points])
            }

            logger.info(f"Loaded gripper calibration data from {calibration_path}")
            logger.info(f"  Target width range: [{self.gripper_calibration_data['target_widths'][0]:.4f}, {self.gripper_calibration_data['target_widths'][-1]:.4f}]")
            logger.info(f"  Measured width range: [{self.gripper_calibration_data['measured_widths'][0]:.4f}, {self.gripper_calibration_data['measured_widths'][-1]:.4f}]")

        except Exception as e:
            logger.error(f"Failed to load gripper calibration data: {e}")
            self.gripper_calibration_data = None

    def _target_to_measured_gripper_width(self, target_width: float) -> float:
        """
        Convert API target width to measured width using calibration data.
        Used in get_robot_state to convert read values.

        Args:
            target_width: Width value read from API

        Returns:
            Corresponding measured width
        """
        if self.gripper_calibration_data is None:
            return target_width

        target_widths = self.gripper_calibration_data["target_widths"]
        measured_widths = self.gripper_calibration_data["measured_widths"]

        # Clip to calibration range
        target_min, target_max = target_widths[0], target_widths[-1]
        clipped_target = np.clip(target_width, target_min, target_max)

        # Find the two nearest points for interpolation
        idx = np.searchsorted(target_widths, clipped_target)

        if idx == 0:
            return measured_widths[0]
        elif idx >= len(target_widths):
            return measured_widths[-1]
        else:
            # Linear interpolation between two nearest points
            t0, t1 = target_widths[idx - 1], target_widths[idx]
            m0, m1 = measured_widths[idx - 1], measured_widths[idx]

            # Interpolation factor
            alpha = (clipped_target - t0) / (t1 - t0) if t1 != t0 else 0.0
            return m0 + alpha * (m1 - m0)

    def _measured_to_target_gripper_width(self, measured_width: float, index: int = 0) -> float:
        """
        Convert desired measured width to API target width using calibration data.
        Used in _execute_eef_command to convert command values.

        Args:
            measured_width: Desired measured width

        Returns:
            Corresponding target width to send to API
        """
        if self.gripper_calibration_data is None:
            return float(measured_width)

        target_widths = self.gripper_calibration_data["target_widths"]
        measured_widths = self.gripper_calibration_data["measured_widths"]

        # Clip to calibration range (measured widths)
        measured_min, measured_max = measured_widths[0], measured_widths[-1]
        clipped_measured = np.clip(measured_width, measured_min, measured_max)

        # Find the two nearest points for interpolation (search in measured_widths)
        idx = np.searchsorted(measured_widths, clipped_measured)

        if idx == 0:
            return float(target_widths[0])
        elif idx >= len(measured_widths):
            return float(target_widths[-1])
        else:
            # Linear interpolation between two nearest points
            m0, m1 = measured_widths[idx - 1], measured_widths[idx]
            t0, t1 = target_widths[idx - 1], target_widths[idx]

            # Interpolation factor
            alpha = (clipped_measured - m0) / (m1 - m0) if m1 != m0 else 0.0
            return float(t0 + alpha * (t1 - t0))
