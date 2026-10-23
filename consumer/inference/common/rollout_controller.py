#!/usr/bin/env python3
"""
Inference rollout process controller for PolicyConnector.

Author: OpenAI Assistant
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from utils.logger_config import logger
from utils.policy_control_config import parse_control_targets

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]


class RolloutController:
    """Manage inference rollout saver lifecycle and recording signals."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def should_start(self) -> bool:
        """Return True when rollout saver is enabled and launch prerequisites exist."""
        owner = self.owner
        if not owner.infer_rollout_config:
            return False
        enabled = owner.infer_rollout_config.get("enabled", False)
        if not enabled:
            return False

        saver_script = self._saver_script()
        if not os.path.exists(saver_script):
            logger.error(f"PolicyConnector: Rollout saver script not found: {saver_script}")
            return False
        if not owner.rollout_python_path:
            logger.error("PolicyConnector: Rollout Python interpreter is not configured")
            return False
        if not os.path.exists(owner.rollout_python_path):
            logger.error(f"PolicyConnector: Python interpreter not found: {owner.rollout_python_path}")
            return False
        return True

    def start(self) -> None:
        """Start the inference rollout data saver process using subprocess."""
        owner = self.owner
        if owner._infer_rollout_started:
            return

        self._precreate_output_shms()
        rollout_cfg = owner.infer_rollout_config
        cmd = self._build_start_command(rollout_cfg)
        ready_file = Path(owner.signal_files.get("rollout_ready"))
        if ready_file.exists():
            ready_file.unlink()
        cmd.extend(["--ready-file", str(ready_file)])

        if rollout_cfg.get("anomaly_config_path"):
            cmd.extend(["--anomaly-config", rollout_cfg.get("anomaly_config_path")])

        cmd.extend(["--signal-files", json.dumps(owner.signal_files)])

        try:
            logger.info(f"PolicyConnector: Starting rollout saver with command: {' '.join(cmd)}")
            owner.infer_rollout_process = subprocess.Popen(
                cmd,
                cwd=str(_PACKAGE_ROOT),
            )
            self._wait_until_ready(ready_file, float(rollout_cfg.get("ready_timeout", 10.0)))
            owner._infer_rollout_started = True
            logger.info(
                f"PolicyConnector: Inference rollout saver started (PID: {owner.infer_rollout_process.pid})"
            )
        except Exception as exc:
            logger.error(f"PolicyConnector: Failed to start inference rollout process: {exc}")
            if owner.infer_rollout_process is not None:
                try:
                    owner.infer_rollout_process.terminate()
                    owner.infer_rollout_process.wait(timeout=2.0)
                except Exception:
                    pass
            owner.infer_rollout_process = None
            owner._infer_rollout_started = False

    def stop(self) -> None:
        """Stop the inference rollout data saver subprocess."""
        owner = self.owner
        if not owner._infer_rollout_started:
            return

        process = owner.infer_rollout_process
        if process is None:
            return

        try:
            if process.poll() is None:
                logger.info("PolicyConnector: Sending SIGINT to inference rollout process for graceful shutdown")
                process.send_signal(signal.SIGINT)

                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    logger.warning("PolicyConnector: Rollout saver did not exit after SIGINT, terminating forcefully")
                    process.terminate()
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        logger.error("PolicyConnector: Rollout saver still alive, force killing")
                        process.kill()
                        process.wait()

            logger.info(f"PolicyConnector: Rollout saver process exited with code {process.returncode}")
        except Exception as exc:
            logger.error(f"PolicyConnector: Error while stopping inference rollout process: {exc}")
        finally:
            owner.infer_rollout_process = None
            owner._infer_rollout_started = False

    def signal_start_recording(self) -> None:
        """Signal the rollout saver to start recording."""
        self._touch_signal_file("start_recording", "start recording")

    def signal_stop_recording(self) -> None:
        """Signal the rollout saver to stop recording."""
        self._touch_signal_file("stop_recording", "stop recording")

    def _precreate_output_shms(self) -> None:
        owner = self.owner
        if owner.value_shm.enabled and not owner.value_shm.created:
            owner.value_shm.create((1,))
            logger.info("PolicyConnector: Pre-created value SHM for rollout saver to connect")

        if owner.action_chunk_shm.enabled:
            action_chunk_shape = owner._get_action_chunk_shape()
            owner.action_chunk_shm.create(action_chunk_shape)
            logger.info(f"PolicyConnector: Pre-created action chunk SHM with shape {action_chunk_shape}")

        q_value_shape = (owner.chunk_length, 1)
        if owner.q_value_shm.enabled:
            owner.q_value_shm.create(q_value_shape)
            logger.info(f"PolicyConnector: Pre-created Q-value SHM with shape {q_value_shape}")
        if owner.q_value_timestamps_shm.enabled:
            owner.q_value_timestamps_shm.create((owner.chunk_length,))
            logger.info(
                "PolicyConnector: Pre-created Q-value timestamps SHM with "
                f"shape ({owner.chunk_length},)"
            )

    def _build_start_command(self, rollout_cfg: dict[str, Any]) -> list[str]:
        owner = self.owner
        if not owner.rollout_python_path:
            raise RuntimeError("Rollout Python interpreter is not configured")
        cmd = [
            owner.rollout_python_path,
            self._saver_script(),
            "--summary-shm",
            rollout_cfg.get("summary_shm_name", "device_summary_data"),
            "--output-dir",
            rollout_cfg.get("output_dir", "./inference_rollout_data"),
            "--interval",
            str(rollout_cfg.get("save_interval", 0.001)),
            "--start-idx",
            str(rollout_cfg.get("start_idx", 0)),
            "--queue-size",
            str(rollout_cfg.get("queue_size", 5000)),
            "--robot-type",
            rollout_cfg.get("robot_type", "Rizon"),
            "--task-name",
            rollout_cfg.get("task_name", rollout_cfg.get("task", "inference_rollout")),
        ]

        image_size = rollout_cfg.get("image_size")
        if image_size:
            cmd.extend(["--image-size", f"{image_size[0]},{image_size[1]}"])

        if rollout_cfg.get("auto_start", True):
            cmd.append("--auto-start")

        control_shms_list = self._build_control_shms_list(rollout_cfg)
        if control_shms_list:
            cmd.extend(["--control-shms", json.dumps(control_shms_list)])
            logger.info(f"PolicyConnector: Adding control SHMs to rollout: {[c['shm'] for c in control_shms_list]}")

        if owner.config_file_path:
            cmd.extend(["--config-file", owner.config_file_path])
            logger.info(f"PolicyConnector: Config file will be saved to rollout: {owner.config_file_path}")

        if rollout_cfg.get("overwrite", False):
            cmd.append("--overwrite")
            logger.info("PolicyConnector: Overwrite mode enabled for rollout")

        return cmd

    def _build_control_shms_list(self, rollout_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        owner = self.owner
        control_shms_list = [
            {
                "shm": target.shm_name,
                "shape": list(target.shape),
                "dtype": target.dtype,
            }
            for target in parse_control_targets(
                rollout_cfg.get("control_shms"),
                field="policy.infer_rollout.control_shms",
            )
        ]

        value_shm_config = owner.value_shm.get_config()
        if owner.value_shm.enabled and owner.value_shm.created and value_shm_config:
            value_shm_config["shape"] = [1]
            control_shms_list.append(value_shm_config)

        for output_shm in (
            owner.action_chunk_shm,
            owner.q_value_shm,
            owner.q_value_timestamps_shm,
        ):
            if not output_shm.enabled or not output_shm.created:
                continue
            output_shm_config = output_shm.get_config()
            if output_shm_config:
                control_shms_list.append(output_shm_config)

        return control_shms_list

    def _wait_until_ready(self, ready_file: Path, rollout_ready_timeout: float) -> None:
        owner = self.owner
        start_time = time.time()
        while time.time() - start_time < rollout_ready_timeout:
            if owner.infer_rollout_process.poll() is not None:
                raise RuntimeError(
                    "Rollout saver process exited unexpectedly with code "
                    f"{owner.infer_rollout_process.returncode}"
                )

            if ready_file.exists():
                content = ready_file.read_text().strip()
                if content == "ready":
                    return

            time.sleep(0.1)
        raise TimeoutError("Inference rollout saver did not signal readiness in time")

    def _touch_signal_file(self, signal_key: str, action_name: str) -> None:
        owner = self.owner
        if not owner._infer_rollout_started:
            return
        try:
            signal_path = Path(owner.signal_files.get(signal_key))
            signal_path.parent.mkdir(parents=True, exist_ok=True)
            signal_path.touch()
            logger.info(f"PolicyConnector: Signaled rollout saver to {action_name}")
        except Exception as exc:
            logger.warning(f"PolicyConnector: Failed to signal {action_name}: {exc}")

    @staticmethod
    def _saver_script() -> str:
        return str(_PACKAGE_ROOT / "consumer" / "io" / "async_infer_data_saver_NEDF3.py")
