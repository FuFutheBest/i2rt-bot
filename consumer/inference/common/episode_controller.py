#!/usr/bin/env python3
"""
Thin episode-control adapter for PolicyConnector.

Author: OpenAI Assistant
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from utils.home_transition import (
    home_result_path,
    normalize_home_target,
)
from utils.logger_config import logger


class EpisodeController:
    """Provide a stable component boundary for reset and episode operations."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def check_reset_signal(self) -> bool:
        """Check if reset signal file exists and has been updated."""
        owner = self.owner
        try:
            reset_signal_file = Path(owner.signal_files.get('reset_signal'))
            if reset_signal_file.exists():
                current_mtime = reset_signal_file.stat().st_mtime
                if current_mtime > owner.last_reset_signal_time:
                    owner.last_reset_signal_time = current_mtime
                    return True
            return False
        except Exception as exc:
            logger.debug(f"Failed to check reset signal: {exc}")
            return False

    def reconnect_all_shms(self) -> None:
        """Delegate SHM reconnection to the current PolicyConnector implementation."""
        self.owner._reconnect_all_shms()

    def signal_policy_runner_reset(self) -> None:
        """Signal the policy runner to clear its action cache and SHM buffers."""
        try:
            signal_path = Path(self.owner.signal_files.get('policy_runner_reset_command'))
            signal_path.parent.mkdir(parents=True, exist_ok=True)
            signal_path.touch()
            logger.info("PolicyConnector: Signaled policy runner to reset")
        except Exception as exc:
            logger.warning(f"PolicyConnector: Failed to signal policy runner reset: {exc}")

    def wait_for_policy_runner_reset(self, timeout_s: Optional[float] = None) -> bool:
        """Wait until the policy runner consumes its reset command."""
        owner = self.owner
        timeout = owner.episode_policy_runner_reset_timeout if timeout_s is None else timeout_s
        signal_path = Path(owner.signal_files.get("policy_runner_reset_command"))
        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            if not signal_path.exists():
                return True
            time.sleep(0.02)
        logger.warning("PolicyConnector: Timed out waiting for policy runner reset")
        return False

    def reset_policy_episode_state(self) -> None:
        """Clear connector and policy state that cannot cross episodes."""
        owner = self.owner
        owner._last_continuous_state_timestamps.clear()
        owner.policy_history_manager.clear_state()
        observation_builder = getattr(owner, "observation_builder", None)
        reset_history = getattr(observation_builder, "reset_history", None)
        if callable(reset_history):
            reset_history()
        if hasattr(owner.policy, "reset"):
            owner.policy.reset()

    def check_and_consume_robot_reconnect_signal(self) -> bool:
        """Consume robot reconnect signal and enqueue episode discard."""
        owner = self.owner
        try:
            reconnect_signal_path = owner.signal_files.get('robot_reconnect')
            if not reconnect_signal_path:
                return False

            reconnect_signal_file = Path(reconnect_signal_path)
            if not reconnect_signal_file.exists():
                return False

            try:
                reconnect_signal_file.unlink()
            except Exception as unlink_err:
                logger.warning(
                    f"PolicyConnector: Failed to remove robot reconnect signal file: {unlink_err}"
                )

            owner._episode_result_queue.put('discard')
            logger.warning(
                "PolicyConnector: Robot reconnect signal detected, enqueueing episode discard "
                "(equivalent to keyboard 'D')."
            )
            logger.info("PolicyConnector: Robot reconnect signal detected, enqueueing episode discard")
            return True
        except Exception as exc:
            logger.warning(f"PolicyConnector: Failed to process robot reconnect signal: {exc}")
            return False

    def check_episode_end_request(self) -> Optional[str]:
        """Return a queued episode result request, if present."""
        try:
            return self.owner._episode_result_queue.get_nowait()
        except queue.Empty:
            return None

    def handle_episode_transition(self, result: str) -> None:
        """Handle episode end: save result, reset, return home, and optionally start new episode."""
        owner = self.owner
        if owner._episode_in_transition:
            logger.warning("PolicyConnector: Episode transition already in progress, ignoring")
            return

        owner._episode_in_transition = True
        logger.info(f"PolicyConnector: ========== EPISODE {owner._episode_count} ENDED: {result.upper()} ==========")

        try:
            self.write_episode_result(result)
            owner._signal_stop_recording()

            self.signal_and_wait_for_robot_halt()
            logger.info(
                "PolicyConnector: Robot halted, sleeping for "
                f"{owner.episode_halt_settle_time}s to avoid state misreading"
            )
            time.sleep(owner.episode_halt_settle_time)

            owner.inference_enabled = False
            self.signal_policy_runner_reset()
            pre_home_reset_ok = self.wait_for_policy_runner_reset()
            self.reset_policy_episode_state()

            home_index = self.read_robot_home_index()
            self.signal_robot_home(home_index=home_index)

            logger.info(f"PolicyConnector: Waiting {owner.episode_home_wait_time}s for robot to return home...")
            time.sleep(owner.episode_home_wait_time)

            self.signal_policy_runner_reset()
            post_home_reset_ok = self.wait_for_policy_runner_reset()
            self.reset_policy_episode_state()
            self.signal_teleoperator_reset()
            owner._episode_count += 1

            if owner.episode_auto_restart and pre_home_reset_ok and post_home_reset_ok:
                logger.info(f"PolicyConnector: ========== STARTING EPISODE {owner._episode_count} ==========")
                owner._signal_start_recording()
                owner.inference_enabled = True
            elif owner.episode_auto_restart:
                logger.error("PolicyConnector: Auto-restart skipped because policy runner reset was not confirmed")
            else:
                logger.info("PolicyConnector: Auto-restart disabled. Waiting for manual restart...")

        except Exception as exc:
            logger.error(f"PolicyConnector: Error during episode transition: {exc}")
        finally:
            owner._episode_in_transition = False

    def write_episode_result(self, result: str) -> None:
        """Write episode result to signal file for data saver to read."""
        try:
            signal_path = Path(self.owner.signal_files.get('episode_result'))
            signal_path.parent.mkdir(parents=True, exist_ok=True)
            signal_path.write_text(result)
            logger.info(f"PolicyConnector: Wrote episode result '{result}' to signal file")
        except Exception as exc:
            logger.error(f"PolicyConnector: Failed to write episode result: {exc}")

    def signal_teleoperator_reset(self) -> None:
        """Signal the teleoperator to switch to policy control mode."""
        try:
            signal_path = Path(self.owner.signal_files.get('teleoperator_reset_command'))
            signal_path.parent.mkdir(parents=True, exist_ok=True)
            signal_path.touch()
            logger.info("PolicyConnector: Signaled teleoperator to reset to policy control mode")
        except Exception as exc:
            logger.warning(f"PolicyConnector: Failed to signal teleoperator reset: {exc}")

    def read_robot_home_index(self) -> int:
        """Read robot home index from shared file written by device_starter.py."""
        try:
            robot_home_index_file = Path(self.owner.signal_files.get('robot_home_index'))
            if robot_home_index_file.exists():
                content = robot_home_index_file.read_text().strip()
                if content:
                    home_index = int(content)
                    logger.debug(f"PolicyConnector: Read robot home index {home_index} from shared file")
                    return home_index
        except Exception as exc:
            logger.warning(f"PolicyConnector: Failed to read robot home index: {exc}")

        logger.debug("PolicyConnector: Using default robot home index 0")
        return 0

    @staticmethod
    def _write_robot_home_command(path: Path, content: str) -> None:
        """Atomically replace a command so workers never read partial JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def signal_robot_home(
        self,
        home_index: int = 0,
        home_target: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Signal all robots in controls config to return to home pose."""
        owner = self.owner
        if not owner.controls_config and not owner.reset_config:
            logger.warning("PolicyConnector: No controls config, cannot signal robot home")
            return False

        normalized_home_target = normalize_home_target(home_target)
        owner._sampled_home_result_files = []
        signaled_devices: set[tuple[str, str]] = set()
        succeeded = True

        for control in owner.controls_config:
            endpoint = control["endpoint"]
            device_name = endpoint.device
            device_class_id = str(endpoint.device_class_id)
            try:
                pattern = owner.signal_files.get('robot_home_command_pattern')
                if not pattern:
                    raise RuntimeError('signal_files.robot_home_command_pattern is missing')
                home_command_file = Path(pattern.format(
                    device_name=device_name, device_class_id=device_class_id
                ))
                command_content = (
                    json.dumps(
                        normalized_home_target,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(',', ':'),
                    )
                    if normalized_home_target is not None and device_name == 'RizonRobot'
                    else str(home_index)
                )
                if normalized_home_target is not None and device_name == 'RizonRobot':
                    result_file = home_result_path(home_command_file)
                    result_file.unlink(missing_ok=True)
                    owner._sampled_home_result_files.append(result_file)
                self._write_robot_home_command(home_command_file, command_content)
                signaled_devices.add((str(device_name), str(device_class_id)))
                logger.info(
                    "PolicyConnector: Signaled {}_{} to {}",
                    device_name,
                    device_class_id,
                    'sampled home candidates'
                    if command_content.startswith('{')
                    else f'return to home pose (index={home_index})',
                )
            except Exception as exc:
                succeeded = False
                logger.error(
                    'PolicyConnector: Failed to signal robot home for {}_{}: {}',
                    device_name,
                    device_class_id,
                    exc,
                )

        for reset_device in owner.reset_config:
            device_name = str(reset_device.get('device_name', ''))
            device_class_id = reset_device.get('device_class_id', 0)
            try:
                if (device_name, str(device_class_id)) in signaled_devices:
                    continue
                pattern = owner.signal_files.get('robot_home_command_pattern')
                if not pattern:
                    raise RuntimeError('signal_files.robot_home_command_pattern is missing')
                reset_command_file = Path(pattern.format(
                    device_name=device_name, device_class_id=device_class_id
                ))
                reset_home_index = reset_device.get('home_index', 0)
                command_content = (
                    json.dumps(
                        normalized_home_target,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(',', ':'),
                    )
                    if normalized_home_target is not None and device_name == 'RizonRobot'
                    else str(reset_home_index)
                )
                if normalized_home_target is not None and device_name == 'RizonRobot':
                    result_file = home_result_path(reset_command_file)
                    result_file.unlink(missing_ok=True)
                    owner._sampled_home_result_files.append(result_file)
                self._write_robot_home_command(
                    reset_command_file,
                    command_content,
                )
                logger.info('PolicyConnector: Signaled {}_{} to reset', device_name, device_class_id)
            except Exception as exc:
                succeeded = False
                logger.error(
                    'PolicyConnector: Failed to signal robot reset for {}_{}: {}',
                    device_name,
                    device_class_id,
                    exc,
                )
        return succeeded and bool(signaled_devices or owner.reset_config)

    def wait_for_sampled_home_results(
        self,
        home_target: Dict[str, Any],
        *,
        timeout_s: float,
    ) -> bool:
        expected = normalize_home_target(home_target)
        pending = set(getattr(self.owner, '_sampled_home_result_files', []))
        if expected is None or not pending:
            logger.error('PolicyConnector: Sampled home result path is unavailable')
            return False
        expected_candidates = {
            side: expected['arms'][side]['candidate_id']
            for side in ('left', 'right')
        }
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        while pending and time.monotonic() < deadline:
            for path in tuple(pending):
                try:
                    info = path.lstat()
                    if path.is_symlink() or not path.is_file() or not 0 < info.st_size <= 65536:
                        raise ValueError('unsafe sampled home result')
                    result = json.loads(path.read_text(encoding='utf-8'))
                except FileNotFoundError:
                    continue
                except Exception as exc:
                    logger.error('PolicyConnector: Invalid sampled home result {}: {}', path, exc)
                    return False
                finally:
                    if path.exists() and not path.is_symlink():
                        path.unlink(missing_ok=True)
                if not isinstance(result, dict):
                    return False
                if (
                    result.get('schema_version') != 1
                    or result.get('config_hash') != expected['config_hash']
                    or result.get('candidates') != expected_candidates
                ):
                    logger.error('PolicyConnector: Sampled home result identity mismatch')
                    return False
                if result.get('status') != 'applied':
                    logger.error(
                        'PolicyConnector: Sampled home failed: {}',
                        result.get('error') or 'unknown device error',
                    )
                    return False
                pending.remove(path)
            if pending:
                time.sleep(0.05)
        if pending:
            logger.error('PolicyConnector: Timed out waiting for sampled home result')
            return False
        return True

    def signal_and_wait_for_robot_halt(self) -> None:
        """Signal robots to halt and wait for their halt reply files."""
        owner = self.owner
        if not self.signal_robot_halt():
            return

        all_robot_halt_reply_files_exist = False
        while not all_robot_halt_reply_files_exist:
            all_robot_halt_reply_files_exist_list = []
            for reply_device in owner.halt_and_home_cfg:
                device_name = reply_device.get('device_name', '')
                device_class_id = reply_device.get('device_class_id', 0)
                robot_halt_reply_file = Path(owner.signal_files.get('robot_halt_reply_pattern').format(
                    device_name=device_name, device_class_id=device_class_id
                ))
                if robot_halt_reply_file.exists():
                    logger.info(f"PolicyConnector: {robot_halt_reply_file} exists")
                    now_robot_halt_reply_file_exists = True
                else:
                    logger.info(f"PolicyConnector: {robot_halt_reply_file} does not exist")
                    now_robot_halt_reply_file_exists = False
                all_robot_halt_reply_files_exist_list.append(now_robot_halt_reply_file_exists)

            all_robot_halt_reply_files_exist = all(all_robot_halt_reply_files_exist_list)

            if all_robot_halt_reply_files_exist:
                break
            time.sleep(0.1)

        logger.info("PolicyConnector: Robot reported its 'do halt state'")

        for reply_device in owner.halt_and_home_cfg:
            device_name = reply_device.get('device_name', '')
            device_class_id = reply_device.get('device_class_id', 0)
            robot_halt_reply_file = Path(owner.signal_files.get('robot_halt_reply_pattern').format(
                device_name=device_name, device_class_id=device_class_id
            ))
            robot_halt_reply_file.unlink()
            logger.info(f"PolicyConnector: Consumed robot halt reply file for {device_name}_{device_class_id}")

    def signal_robot_halt(self) -> bool:
        """Signal configured robots to halt without waiting for an acknowledgement."""
        owner = self.owner
        try:
            for halt_device in owner.halt_and_home_cfg:
                device_name = halt_device.get('device_name', '')
                device_class_id = halt_device.get('device_class_id', 0)
                robot_halt_signal = Path(owner.signal_files.get('robot_halt_command_pattern').format(
                    device_name=device_name, device_class_id=device_class_id
                ))
                robot_halt_signal.parent.mkdir(parents=True, exist_ok=True)
                robot_halt_signal.touch()
                logger.info(f"PolicyConnector: Signaled robot to halt, device: {device_name}_{device_class_id}")
        except Exception as exc:
            logger.error(f"PolicyConnector: Failed to signal robot halt: {exc}")
            return False
        return True
