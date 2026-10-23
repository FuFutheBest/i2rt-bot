#!/usr/bin/env python3
"""Rizon robot helper mixin.

Author: OpenAI Assistant
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from utils.home_transition import (
    home_result_path,
    normalize_home_target,
)
from utils.logger_config import logger
from utils.shm_utils import (
    get_control_frame_offset,
)


class RizonHomeMixin:
    """Focused RizonRobot helper methods from home.py."""

    def _configure_home(self, home_cfg: Dict[str, Any]) -> None:
        """Configure home configuration from YAML dict."""
        try:
            if self.is_dual_arm:
                self._configure_dual_arm_home(home_cfg)
            else:
                self._configure_single_arm_home(home_cfg)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.error(f"Failed to configure home parameters: {exc}")
            raise

    def _configure_dual_arm_home(self, home_cfg: Dict[str, Any]) -> None:
        """Configure dual-arm home joints and poses."""
        presets = home_cfg.get("presets")
        preset_index = home_cfg.get("preset_index")

        if presets is not None:
            if not isinstance(presets, list) or len(presets) == 0:
                raise ValueError("home.presets for dual-arm configuration must be a non-empty list")
            self.home_presets = presets

            if preset_index is None:
                preset_index = 0
            if not isinstance(preset_index, int):
                raise TypeError("home.preset_index must be an integer")
            if not (0 <= preset_index < len(presets)):
                raise IndexError(f"home.preset_index {preset_index} is out of bounds for {len(presets)} presets")

            self.home_preset_index = preset_index
            selected_preset = presets[preset_index]

            if not isinstance(selected_preset, dict):
                raise TypeError(f"Preset at index {preset_index} must be a mapping")

            joints_left = selected_preset.get("joints_left")
            joints_right = selected_preset.get("joints_right")
            pose_left = selected_preset.get("pose_left")
            pose_right = selected_preset.get("pose_right")

            if joints_left is not None:
                self.home_joints_left = joints_left
                logger.info(f"Configured home joints for left arm (preset {preset_index}): {self.home_joints_left}")
            if joints_right is not None:
                self.home_joints_right = joints_right
                logger.info(f"Configured home joints for right arm (preset {preset_index}): {self.home_joints_right}")
            if pose_left is not None:
                self.home_pose_left = pose_left
                logger.info(f"Configured home pose for left arm (preset {preset_index}): {self.home_pose_left}")
            if pose_right is not None:
                self.home_pose_right = pose_right
                logger.info(f"Configured home pose for right arm (preset {preset_index}): {self.home_pose_right}")
        else:
            if "preset_index" in home_cfg:
                raise KeyError("home.preset_index specified without defining home.presets")

            if "joints_left" in home_cfg:
                self.home_joints_left = home_cfg["joints_left"]
                logger.info(f"Configured home joints for left arm: {self.home_joints_left}")
            if "joints_right" in home_cfg:
                self.home_joints_right = home_cfg["joints_right"]
                logger.info(f"Configured home joints for right arm: {self.home_joints_right}")
            if "pose_left" in home_cfg:
                self.home_pose_left = home_cfg["pose_left"]
                logger.info(f"Configured home pose for left arm: {self.home_pose_left}")
            if "pose_right" in home_cfg:
                self.home_pose_right = home_cfg["pose_right"]
                logger.info(f"Configured home pose for right arm: {self.home_pose_right}")

    def _configure_single_arm_home(self, home_cfg: Dict[str, Any]) -> None:
        """Configure single-arm home joints and pose."""
        presets = home_cfg.get("presets")
        preset_index = home_cfg.get("preset_index")

        if presets is not None:
            if not isinstance(presets, list) or len(presets) == 0:
                raise ValueError("home.presets must be a non-empty list")

            self.home_presets = presets

            if preset_index is None:
                preset_index = 0
            if not isinstance(preset_index, int):
                raise TypeError("home.preset_index must be an integer")
            if not (0 <= preset_index < len(presets)):
                raise IndexError(f"home.preset_index {preset_index} is out of bounds for {len(presets)} presets")

            self.home_preset_index = preset_index
            selected_preset = presets[preset_index]

            if not isinstance(selected_preset, dict):
                raise TypeError(f"Preset at index {preset_index} must be a mapping")

            if "joints" in selected_preset:
                self.home_joints = selected_preset["joints"]
                logger.info(f"Configured home joints (preset {preset_index}): {self.home_joints}")
            if "pose" in selected_preset:
                self.home_pose = selected_preset["pose"]
                logger.info(f"Configured home pose (preset {preset_index}): {self.home_pose}")
        else:
            if "preset_index" in home_cfg:
                raise KeyError("home.preset_index specified without defining home.presets")

            if "joints" in home_cfg:
                self.home_joints = home_cfg["joints"]
                logger.info(f"Configured home joints: {self.home_joints}")
            if "pose" in home_cfg:
                self.home_pose = home_cfg["pose"]
                logger.info(f"Configured home pose: {self.home_pose}")

    def _set_home_pose(self) -> bool:
        """Set home pose from config or current pose if not configured."""
        try:
            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right:
                    return False

                # If home poses not configured, use current poses
                if self.home_pose_left is None or self.home_pose_right is None:
                    logger.info("Home poses not configured, using current poses")

                    # Check if home_joints are configured - if so, move there first and record poses
                    if self.home_joints_left is not None or self.home_joints_right is not None:
                        logger.info("Moving to configured home joint positions")
                        try:
                            # Ensure joint control mode before sending joint positions
                            return_to_eef = self.control_type in ["eef_abs", "eef_rel"]
                            if not self._is_joint_mode_active():
                                if not self._switch_to_joint_control():
                                    raise RuntimeError("Failed to switch to joint position control before homing")
                            # Move left arm to home joint positions if configured
                            if self.home_joints_left is not None:
                                logger.info(f"Moving left arm to home joints: {self.home_joints_left}")
                                self._send_joint_position_command(
                                    self.robot_left,
                                    self.home_joints_left,
                                    self.max_velocity[:7],
                                    self.max_acceleration[:7],
                                )

                            # Move right arm to home joint positions if configured
                            if self.home_joints_right is not None:
                                logger.info(f"Moving right arm to home joints: {self.home_joints_right}")
                                self._send_joint_position_command(
                                    self.robot_right,
                                    self.home_joints_right,
                                    self.max_velocity[7:14],
                                    self.max_acceleration[7:14],
                                )

                            # Wait for motion to complete
                            max_wait_time = 10.0
                            start_time = time.time()
                            while time.time() - start_time < max_wait_time:
                                left_reached = right_reached = False
                                if self.home_joints_left is not None and np.allclose(self.robot_left.states().q, self.home_joints_left, atol=1e-3):
                                    left_reached = True
                                if self.home_joints_right is not None and np.allclose(self.robot_right.states().q, self.home_joints_right, atol=1e-3):
                                    right_reached = True
                                if left_reached and right_reached:
                                    break
                                time.sleep(0.1)
                            logger.info("Dual-arm robots moved to home joint positions")

                            # Reset gripper widths to home position (same as initialization)
                            if self.utilize_gripper and self.gripper_left:
                                self._move_gripper_home_by_backend(self.gripper_left, 0, "left", wait=True)
                            if self.utilize_gripper and self.gripper_right:
                                self._move_gripper_home_by_backend(self.gripper_right, 1, "right", wait=True)

                            # Now get the poses at these joint configurations and set as home poses
                            left_states = self.robot_left.states()
                            right_states = self.robot_right.states()
                            left_tcp = left_states.tcp_pose.copy()
                            right_tcp = right_states.tcp_pose.copy()

                            # Convert from Flexiv format [x,y,z,qw,qx,qy,qz] to standard [x,y,z,qx,qy,qz,qw]
                            if self.home_pose_left is None:
                                self.home_pose_left = self._flexiv_pose_to_xyzw_pose(left_tcp)
                                logger.info(f"Recorded home pose for left arm from home joints: pos={self.home_pose_left[:3]}, quat={self.home_pose_left[3:]}")

                            if self.home_pose_right is None:
                                self.home_pose_right = self._flexiv_pose_to_xyzw_pose(right_tcp)
                                logger.info(f"Recorded home pose for right arm from home joints: pos={self.home_pose_right[:3]}, quat={self.home_pose_right[3:]}")

                            # Switch back to EEF control if needed
                            if return_to_eef and not self._is_eef_mode_active():
                                if not self._switch_to_eef_control():
                                    logger.warning("Failed to switch back to EEF control after homing")

                        except Exception as e:
                            logger.error(f"Failed to move to home joints: {e}")
                            # Fall back to using current poses
                            logger.info("Falling back to using current poses as home poses")
                            left_states = self.robot_left.states()
                            right_states = self.robot_right.states()
                            left_tcp = left_states.tcp_pose.copy()
                            right_tcp = right_states.tcp_pose.copy()

                            if self.home_pose_left is None:
                                self.home_pose_left = self._flexiv_pose_to_xyzw_pose(left_tcp)
                            if self.home_pose_right is None:
                                self.home_pose_right = self._flexiv_pose_to_xyzw_pose(right_tcp)

                    else:
                        logger.info(f"Current left joints: {self.robot_left.states().q}")
                        logger.info(f"Current right joints: {self.robot_right.states().q}")

                    # If still no home poses, get current poses
                    if self.home_pose_left is None or self.home_pose_right is None:
                        # Get current poses for both arms
                        left_states = self.robot_left.states()
                        right_states = self.robot_right.states()

                        # Set home poses (TCP pose: [x, y, z, qw, qx, qy, qz] -> [x, y, z, qx, qy, qz, qw])
                        left_tcp = left_states.tcp_pose.copy()
                        right_tcp = right_states.tcp_pose.copy()

                        if self.home_pose_left is None:
                            # Convert from Flexiv format [x,y,z,qw,qx,qy,qz] to standard [x,y,z,qx,qy,qz,qw]
                            self.home_pose_left = self._flexiv_pose_to_xyzw_pose(left_tcp)

                        if self.home_pose_right is None:
                            # Convert from Flexiv format [x,y,z,qw,qx,qy,qz] to standard [x,y,z,qx,qy,qz,qw]
                            self.home_pose_right = self._flexiv_pose_to_xyzw_pose(right_tcp)

                logger.info("Using dual-arm home poses:")
                logger.info(f"  Left: pos={self.home_pose_left[:3]}, quat={self.home_pose_left[3:]}")
                logger.info(f"  Right: pos={self.home_pose_right[:3]}, quat={self.home_pose_right[3:]}")
            else:
                if not self.robot:
                    return False

                # Check if home_joints is configured - if so, move there first and record pose
                if self.home_joints is not None:
                    logger.info(f"Moving to configured home joints: {self.home_joints}")
                    try:
                        # Ensure joint control mode before sending joint positions
                        return_to_eef = self.control_type in ["eef_abs", "eef_rel"]
                        if not self._is_joint_mode_active():
                            if not self._switch_to_joint_control():
                                raise RuntimeError("Failed to switch to joint position control before homing")

                        # Move to home joint positions
                        self._send_joint_position_command(
                            self.robot,
                            self.home_joints,
                            self.max_velocity,
                            self.max_acceleration,
                        )
                        # Wait for motion to complete
                        max_wait_time = 10.0
                        start_time = time.time()
                        while time.time() - start_time < max_wait_time:
                            reached = False
                            if self.home_joints is not None and np.allclose(self.robot.states().q, self.home_joints, atol=1e-3):
                                reached = True
                            if reached:
                                break
                            time.sleep(0.1)
                        logger.info("Robot moved to home joint positions")

                        # Reset gripper width to home position (same as initialization)
                        if self.utilize_gripper and self.gripper:
                            self._move_gripper_home_by_backend(self.gripper, 0, "single", wait=True)

                        # Now get the pose at this joint configuration and set as home_pose
                        states = self.robot.states()
                        tcp = states.tcp_pose.copy()

                        # Convert from Flexiv format [x,y,z,qw,qx,qy,qz] to standard [x,y,z,qx,qy,qz,qw]
                        self.home_pose = self._flexiv_pose_to_xyzw_pose(tcp)
                        logger.info(f"Recorded home pose from home joints: pos={self.home_pose[:3]}, quat={self.home_pose[3:]}")

                        # Switch back to EEF control if needed
                        if return_to_eef and not self._is_eef_mode_active():
                            if not self._switch_to_eef_control():
                                logger.warning("Failed to switch back to EEF control after homing")

                    except Exception as e:
                        logger.error(f"Failed to move to home joints: {e}")
                        # Fall back to using current pose
                        logger.info("Falling back to using current pose as home pose")
                        states = self.robot.states()
                        tcp = states.tcp_pose.copy()
                        self.home_pose = self._flexiv_pose_to_xyzw_pose(tcp)

                else:
                    logger.info(f"Current joints: {self.robot.states().q}")

                # If home pose not configured, use current pose
                if self.home_pose is None:
                    logger.info("Home pose not configured, using current pose")

                    # Get current pose for single arm
                    states = self.robot.states()
                    tcp = states.tcp_pose.copy()

                    # Convert from Flexiv format [x,y,z,qw,qx,qy,qz] to standard [x,y,z,qx,qy,qz,qw]
                    self.home_pose = self._flexiv_pose_to_xyzw_pose(tcp)

                logger.info(f"Using home pose: pos={self.home_pose[:3]}, quat={self.home_pose[3:]}")

            return True

        except Exception as e:
            logger.error(f"Failed to set home pose: {e}")
            return False

    def _publish_live_robot_state(self) -> None:
        """Keep the state SHM current while synchronous Home motion is active."""
        if getattr(self, "shared_memory", None) is None:
            return
        state_array = self.get_robot_state()
        if state_array is not None:
            self._write_array_to_shm_with_timestamp(state_array, time.time_ns())

    def _go_home_with_candidate(self, home_target: Dict[str, Any]) -> bool:
        """Move both Rizon arms to one validated sampled joint target."""
        target = normalize_home_target(home_target)
        if target is None or not self.is_dual_arm:
            logger.error("Sampled home target requires one dual-arm Rizon device")
            return False
        if not self.robot_left or not self.robot_right:
            logger.error("Dual-arm robots are unavailable for sampled home motion")
            return False

        left_target = np.asarray(target["arms"]["left"]["joints"], dtype=float)
        right_target = np.asarray(target["arms"]["right"]["joints"], dtype=float)
        return_to_eef = self.control_type in {"eef_abs", "eef_rel"}
        try:
            if not self._is_joint_mode_active() and not self._switch_to_joint_control():
                raise RuntimeError("Failed to enter joint mode for sampled home motion")
            logger.info(
                "Moving dual-arm Rizon to sampled home candidates {}/{}",
                target["arms"]["left"]["candidate_id"],
                target["arms"]["right"]["candidate_id"],
            )
            self._send_joint_position_command(
                self.robot_left, left_target, self.max_velocity[:7], self.max_acceleration[:7]
            )
            self._send_joint_position_command(
                self.robot_right, right_target, self.max_velocity[7:14], self.max_acceleration[7:14]
            )
            deadline = time.monotonic() + 12.0
            reached = False
            while time.monotonic() < deadline:
                left_states = self.robot_left.states()
                right_states = self.robot_right.states()
                reached = bool(
                    np.allclose(left_states.q, left_target, atol=0.01, rtol=0.0)
                    and np.allclose(right_states.q, right_target, atol=0.01, rtol=0.0)
                )
                self._publish_live_robot_state()
                if reached:
                    break
                time.sleep(0.05)
            if not reached:
                raise RuntimeError("Sampled home joint motion timed out")

            if self.utilize_gripper and self.gripper_left:
                self._move_gripper_home_by_backend(self.gripper_left, 0, "left", wait=False)
            if self.utilize_gripper and self.gripper_right:
                self._move_gripper_home_by_backend(self.gripper_right, 1, "right", wait=False)
            self.home_pose_left = self._flexiv_pose_to_xyzw_pose(
                self.robot_left.states().tcp_pose.copy()
            )
            self.home_pose_right = self._flexiv_pose_to_xyzw_pose(
                self.robot_right.states().tcp_pose.copy()
            )
            if return_to_eef and not self._switch_to_eef_control():
                raise RuntimeError("Failed to restore EEF mode after sampled home motion")
            self._publish_live_robot_state()
            return True
        except Exception as exc:
            logger.error("Failed sampled home motion: {}", exc)
            return False
        finally:
            if return_to_eef and not self._is_eef_mode_active():
                if not self._switch_to_eef_control():
                    logger.error("Could not restore EEF mode after sampled home failure")

    @staticmethod
    def _write_sampled_home_result(
        path: Path,
        status: str,
        home_target: Optional[Dict[str, Any]],
        *,
        error: Optional[str] = None,
    ) -> None:
        target = normalize_home_target(home_target)
        payload = {
            "schema_version": 1,
            "status": status,
            "config_hash": target.get("config_hash") if target else None,
            "candidates": (
                {side: target["arms"][side]["candidate_id"] for side in ("left", "right")}
                if target
                else None
            ),
            "completed_at_ns": time.time_ns(),
            "error": error,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _check_and_execute_home_command(self) -> None:
        """Consume an integer preset or a validated sampled-joint Home command."""
        try:
            home_command_pattern = self._robot_home_command_pattern
            assert home_command_pattern is not None, "home_command_pattern is required in signal_files"
            home_command_file = Path(home_command_pattern.format(device_name=self.device_name, device_class_id=self.device_class_id))
            if not home_command_file.exists():
                return

            sampled_command = False
            home_target: Optional[Dict[str, Any]] = None
            try:
                content = home_command_file.read_text(encoding="utf-8").strip()
                if content.startswith("{"):
                    sampled_command = True
                    home_target = normalize_home_target(json.loads(content))
                    if home_target is None:
                        raise ValueError("Sampled home command is empty")
                    home_idx = None
                else:
                    home_idx = int(content) if content else (
                        self.home_preset_index if self.home_preset_index is not None else 0
                    )
            except Exception as exc:
                home_command_file.unlink(missing_ok=True)
                if sampled_command:
                    self._write_sampled_home_result(
                        home_result_path(home_command_file), "failed", home_target,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                logger.error("Error parsing external home command: {}", exc)
                return

            home_command_file.unlink(missing_ok=True)
            self._clear_control_shm()
            success = (
                self._go_home_with_candidate(home_target)
                if home_target is not None
                else self._go_home_with_index(int(home_idx))
            )
            self._clear_control_shm()
            if not success:
                if home_target is not None:
                    self._write_sampled_home_result(
                        home_result_path(home_command_file), "failed", home_target,
                        error="Rizon home command failed",
                    )
                logger.error("Failed to execute external home command")
                return

            # Reject actions generated before the Home barrier once motion has
            # completed. This keeps a late policy frame from immediately
            # pulling the robot back toward the previous episode's pose.
            home_end_timestamp_ns = time.time_ns()
            self._last_home_command_timestamp = max(
                getattr(self, "_last_home_command_timestamp", 0),
                home_end_timestamp_ns,
            )
            self._policy_control_start_timestamp = max(
                getattr(self, "_policy_control_start_timestamp", 0),
                home_end_timestamp_ns,
            )
            if home_target is not None:
                self._write_sampled_home_result(
                    home_result_path(home_command_file), "applied", home_target
                )
            self._notify_policy_runner_reset()
            logger.info("Successfully executed external home command")
        except Exception as e:
            logger.debug(f"Error checking home command: {e}")

    def _notify_policy_runner_reset(self) -> None:
        """Notify policy runner and policy connector to reset when robot is resetting."""
        try:
            from pathlib import Path

            # Create reset command file for policy runner (ActionExecutor)
            reset_command_file = Path(self._policy_runner_reset_command)
            reset_command_file.parent.mkdir(parents=True, exist_ok=True)
            reset_command_file.touch()

            # Also create reset signal file for policy connector
            # This notifies the policy connector to reset its RTC state
            reset_signal_file = Path(self._reset_signal)
            reset_signal_file.touch()

            logger.info("Notified policy runner and policy connector to reset due to robot reset")
        except Exception as e:
            logger.warning(f"Failed to notify policy runner/connector of reset: {e}")

    def _clear_control_shm(self) -> None:
        """Clear all data in robot control SHM buffers."""
        try:
            from utils.shm_utils import (
                BUFFER_HEADER_SIZE,
                DEVICE_HEADER_SIZE,
                FRAME_HEADER_SIZE,
                connect_to_shared_memory,
                pack_buffer_header,
                pack_frame_header,
                unpack_buffer_header,
                unpack_device_header,
            )

            # A source endpoint may be owned by another process.  Homing may
            # invalidate this robot's policy command, never a producer's state.
            for shm_name in self._owned_control_shm_names:
                control_shm = self.control_shared_memory.get(shm_name)
                if control_shm is None:
                    continue
                write_shm = None
                buf = None
                try:
                    channel = control_shm
                    if getattr(control_shm, "is_read_only", False):
                        write_shm = connect_to_shared_memory(shm_name, read_only=False)
                        channel = write_shm
                    buf = channel.buf

                    # Read buffer header
                    buffer_header = unpack_buffer_header(buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE+BUFFER_HEADER_SIZE])
                    buffer_size = buffer_header['buffer_size']

                    # Read device header to get frame_size
                    device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
                    frame_size = device_header['frame_size']

                    # Clear all frames by setting timestamps to 0
                    for i in range(buffer_size):
                        frame_offset = get_control_frame_offset(i, frame_size)
                        # Set frame timestamp to 0 (invalid timestamp)
                        zero_timestamp_header = pack_frame_header(0)
                        buf[frame_offset:frame_offset+FRAME_HEADER_SIZE] = zero_timestamp_header
                        # Clear frame data (set to zeros)
                        data_start = frame_offset + FRAME_HEADER_SIZE
                        data_end = frame_offset + frame_size
                        buf[data_start:data_end] = b'\x00' * (frame_size - FRAME_HEADER_SIZE)

                    # Reset buffer header (write_index=0, current_frames_count=0)
                    reset_buffer_header = pack_buffer_header(buffer_size, 0, 0)
                    buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE+BUFFER_HEADER_SIZE] = reset_buffer_header

                    logger.info(f"Cleared control SHM buffer: {shm_name}")
                except Exception as e:
                    logger.error(f"Failed to clear control SHM {shm_name}: {e}")
                finally:
                    del buf
                    if write_shm is not None:
                        write_shm.close()
            if self.use_cartesian_motion_planner:
                self._reset_planner_command_timeline()
        except Exception as e:
            logger.error(f"Error clearing control SHM: {e}")

    def _go_home_with_index(self, home_idx: int) -> bool:
        """Move robot(s) to home pose with specified preset index."""
        try:
            # Update home preset index if needed
            if self.home_presets is not None and 0 <= home_idx < len(self.home_presets):
                self.home_preset_index = home_idx
                selected_preset = self.home_presets[home_idx]

                # Update home joints/poses from preset
                # Priority: if joints are configured, use joints (will record pose from joints)
                # Otherwise, use pose directly
                if self.is_dual_arm:
                    if "joints_left" in selected_preset:
                        self.home_joints_left = selected_preset["joints_left"]
                        # Clear pose_left to force _set_home_pose to use joints and record pose
                        self.home_pose_left = None
                    if "joints_right" in selected_preset:
                        self.home_joints_right = selected_preset["joints_right"]
                        # Clear pose_right to force _set_home_pose to use joints and record pose
                        self.home_pose_right = None
                    # Only set pose if joints are not available
                    if "pose_left" in selected_preset and self.home_joints_left is None:
                        self.home_pose_left = selected_preset["pose_left"]
                    if "pose_right" in selected_preset and self.home_joints_right is None:
                        self.home_pose_right = selected_preset["pose_right"]
                else:
                    if "joints" in selected_preset:
                        self.home_joints = selected_preset["joints"]
                        # Clear pose to force _set_home_pose to use joints and record pose
                        self.home_pose = None
                    # Only set pose if joints are not available
                    if "pose" in selected_preset and self.home_joints is None:
                        self.home_pose = selected_preset["pose"]

            # Set home pose from config or current pose (same as initialization)
            # This ensures home poses are properly set, especially if using joints
            # If joints are configured, this will move to joints first and record poses
            if not self._set_home_pose():
                logger.error("Failed to set home pose before going home")
                return False

            # Call the original _go_home method
            return self._go_home()
        except Exception as e:
            logger.error(f"Failed to go home with index {home_idx}: {e}")
            return False

    def _go_home(self) -> bool:
        """Move robot(s) to home pose."""
        try:
            if self.control_type == "joints":
                logger.info("Joint control mode - skipping go_home")
                return True

            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right or not self.home_pose_left or not self.home_pose_right:
                    logger.error("Dual-arm robots or home poses not initialized")
                    return False

                # Move both arms to home poses
                # Convert back to Flexiv format [x,y,z,qw,qx,qy,qz]
                left_target = self._xyzw_pose_to_flexiv_pose(self.home_pose_left)
                right_target = self._xyzw_pose_to_flexiv_pose(self.home_pose_right)

                logger.info("Moving dual-arm robots to home poses...")
                self._send_cartesian_motion_command(self.robot_left, left_target)
                self._send_cartesian_motion_command(self.robot_right, right_target)

                # Reset gripper widths to home position (same as initialization)
                if self.utilize_gripper and self.gripper_left:
                    self._move_gripper_home_by_backend(self.gripper_left, 0, "left", wait=False)
                if self.utilize_gripper and self.gripper_right:
                    self._move_gripper_home_by_backend(self.gripper_right, 1, "right", wait=False)

                # Wait for motion to complete
                time.sleep(3.0)
                logger.info("Dual-arm robots moved to home poses")
            else:
                if not self.robot or not self.home_pose:
                    logger.error("Robot or home pose not initialized")
                    return False

                # Move single arm to home pose
                # Convert back to Flexiv format [x,y,z,qw,qx,qy,qz]
                target = self._xyzw_pose_to_flexiv_pose(self.home_pose)

                logger.info("Moving robot to home pose...")
                self._send_cartesian_motion_command(self.robot, target)

                # Reset gripper width to home position (same as initialization)
                if self.utilize_gripper and self.gripper:
                    self._move_gripper_home_by_backend(self.gripper, 0, "single", wait=False)

                # Wait for motion to complete
                time.sleep(3.0)
                logger.info("Robot moved to home pose")

            return True

        except Exception as e:
            logger.error(f"Failed to go home: {e}")
            return False
