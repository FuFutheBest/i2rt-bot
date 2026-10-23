#!/usr/bin/env python3
"""
Asynchronous inference data saver for canonical NEDF3 rollout episodes.

This module provides data saving functionality for inference rollout by consuming device manager data
and storing it with the R1/Flexiv NEDF3 HAL contract. It is controlled via signal files.

Author: Dong Jiulong, Modified for inference rollout
"""

import sys
import time
import signal
import argparse
import os
import json
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import queue
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nmx_msg.Lowdim_pb2 import LowdimData
from nmx_msg.Image_pb2 import Image

from utils.logger_config import logger
from utils.time_control import precise_loop_timing
from utils.shm_utils import (
    connect_to_shared_memory,
    DEVICE_HEADER_SIZE, BUFFER_HEADER_SIZE, FRAME_HEADER_SIZE,
    unpack_device_header, unpack_buffer_header, get_dtype
)

from incubator.r1.async_data_saver_NEDF3_r1 import R1DataSaverConsumer as BaseDataSaverConsumer
from utils.recording_control import (
    RecordingControlPlane,
    write_conversion_source_receipt,
)


class InferenceDataSaverConsumer(BaseDataSaverConsumer):
    """
    Data saver consumer for inference rollout - controlled via signal files.

    This consumer saves data using NEDF SDK during inference:
    - Camera devices: Saved as JPG compressed images in MCAP format
    - Robot devices: Saved as lowdim data (TCP, Joint, etc.)
    - Other devices: Saved as custom lowdim topics
    - Controlled via signal files instead of keyboard input
    - Supports auto_start mode to begin recording immediately
    """

    def __init__(self,
                 summary_shm_name: str = "device_summary_data",
                 output_dir: str = "inference_rollout_data",
                 save_interval: float = 0.04,
                 start_idx: int = 0,
                 image_size: Optional[Tuple[int, int]] = None,
                 async_write: bool = True,
                 queue_size: int = 5000,
                 robot_type: str = "Rizon",
                 task_name: str = "inference_rollout",
                 ready_conn = None,
                 anomaly_config_path: Optional[str] = None,
                 auto_start: bool = False,
                 control_shms: Optional[List[Dict[str, Any]]] = None,
                 signal_files: Optional[Dict[str, str]] = None) -> None:
        """
        Initialize the inference data saver consumer.

        Args:
            summary_shm_name: Name of the summary shared memory
            output_dir: Directory to save data files
            save_interval: Interval between saves in seconds
            start_idx: Starting episode index
            image_size: Target image size (width, height) for resizing
            async_write: Whether to use async writing thread (default: True)
            queue_size: Size of async write queue (default: 5000)
            robot_type: Robot type for NEDF metadata
            task_name: Task name for NEDF metadata
            ready_conn: Connection for sending ready signal to parent process
            anomaly_config_path: Path to anomaly detection config file
            auto_start: Whether to start recording immediately (default: False)
            control_shms: List of control SHM configurations, each with 'shm', 'shape', 'dtype'
            signal_files: Dictionary of signal file paths
        """
        # Build signal_files dict if not provided
        if signal_files is None:
            raise ValueError("signal_files must be provided in config")

        # Validate required signal files for inference
        required_signals = ['start_recording', 'stop_recording', 'episode_result', 'robot_home_command_pattern', 'robot_teleop_start_command_pattern', 'episode_signal_pattern']
        for sig in required_signals:
            if sig not in signal_files:
                raise ValueError(f"signal_files.{sig} is required in config")

        # Pass signal_files to parent class
        super().__init__(
            summary_shm_name=summary_shm_name,
            output_dir=output_dir,
            save_interval=save_interval,
            start_idx=start_idx,
            image_size=image_size,
            async_write=async_write,
            queue_size=queue_size,
            robot_type=robot_type,
            task_name=task_name,
            anomaly_config_path=anomaly_config_path,
            signal_files=signal_files
        )

        self.ready_conn = ready_conn
        self.auto_start = auto_start

        # Use signal files from config for inference control
        self.start_recording_signal = Path(signal_files.get('start_recording'))
        self.stop_recording_signal = Path(signal_files.get('stop_recording'))
        self.episode_result_signal = Path(signal_files.get('episode_result'))
        self.recording_active = False
        self.episode_dir = None
        self.current_episode_result: Optional[str] = None  # 'success', 'failure', or None

        # Control SHM configuration
        self.control_shm_configs: List[Dict[str, Any]] = control_shms or []
        self.control_shm_connections: Dict[str, Any] = {}  # shm_name -> {shm, shape, dtype, last_timestamp}

        # Control Panel owns the run directory and waits for this recorder to
        # publish ``armed`` before allowing inference to become ready. Native
        # launches do not set these environment variables, so they retain the
        # legacy signal-file behavior.
        reference_name = next(
            (
                str(config.get("shm"))
                for config in self.control_shm_configs
                if isinstance(config, dict)
                and str(config.get("shm", "")).endswith("_control")
            ),
            None,
        )
        self.recording_control = RecordingControlPlane.from_environment(
            reference_name=reference_name
        )

        # Clean up any existing signal files
        self._cleanup_signal_files()

    def _connect_control_shms(self) -> bool:
        """
        Connect to control shared memories specified in config.
        Reads device header from SHM to get actual shape, dtype, and frame_size.

        Returns:
            True if all connections successful, False otherwise
        """
        if not self.control_shm_configs:
            return True

        for cfg in self.control_shm_configs:
            shm_name = cfg.get('shm')

            if not shm_name:
                continue

            try:
                # Connect to control SHM in read-only mode
                control_shm = connect_to_shared_memory(shm_name, read_only=True)

                # Read device header from SHM to get actual parameters
                # This is how robot.py validates and reads control SHM
                device_header_data = bytes(control_shm.buf[:DEVICE_HEADER_SIZE])
                device_info = unpack_device_header(device_header_data)

                # Extract actual values from device header
                shape = device_info['shape']
                dtype = device_info['data_dtype']
                frame_size = device_info['frame_size']  # This is total frame size including header
                frame_data_size = frame_size - FRAME_HEADER_SIZE

                # Read buffer header to get buffer_size
                buffer_header_data = bytes(control_shm.buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE])
                buffer_info = unpack_buffer_header(buffer_header_data)
                buffer_size = buffer_info['buffer_size']

                # Pre-calculate frame offsets (like robot.py does)
                frame_offsets = [
                    DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + i * frame_size
                    for i in range(buffer_size)
                ]

                self.control_shm_connections[shm_name] = {
                    'shm': control_shm,
                    'shape': shape,
                    'dtype': dtype,
                    'last_timestamp': 0,
                    'frame_size': frame_size,
                    'frame_data_size': frame_data_size,
                    'buffer_size': buffer_size,
                    'frame_offsets': frame_offsets
                }

                self._write_debug_log(f"Connected to control SHM: {shm_name}")
                self._write_debug_log(f"  - shape={shape}, dtype={dtype}")
                self._write_debug_log(f"  - frame_size={frame_size}, frame_data_size={frame_data_size}")
                self._write_debug_log(f"  - buffer_size={buffer_size}")
                logger.info(f"Connected to control SHM: {shm_name}")

            except FileNotFoundError:
                logger.warning(f"Control SHM not found: {shm_name}, skipping")
                self._write_debug_log(f"Control SHM not found: {shm_name}")
            except Exception as e:
                logger.error(f"Failed to connect to control SHM {shm_name}: {e}")
                self._write_debug_log(f"Failed to connect to control SHM {shm_name}: {e}")
                import traceback
                self._write_debug_log(traceback.format_exc())

        return len(self.control_shm_connections) > 0 or not self.control_shm_configs

    def _read_control_shm_data(self, shm_name: str) -> Optional[Tuple[int, np.ndarray]]:
        """
        Read latest data from a control shared memory.

        This follows the exact same structure as robot.py's read_latest_control_command method.
        Control SHM structure: device_header + buffer_header + frames

        Args:
            shm_name: Name of the control SHM

        Returns:
            Tuple of (timestamp_ns, data_array) or None if read failed
        """
        if shm_name not in self.control_shm_connections:
            return None

        conn_info = self.control_shm_connections[shm_name]
        control_shm = conn_info['shm']
        shape = conn_info['shape']
        dtype = conn_info['dtype']
        frame_data_size = conn_info['frame_data_size']
        buffer_size = conn_info['buffer_size']
        frame_offsets = conn_info['frame_offsets']

        try:
            buf = control_shm.buf

            # Read buffer header to get current write position
            # Same as robot.py's _get_control_buffer_info
            buffer_header_data = bytes(buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE])
            buffer_info = unpack_buffer_header(buffer_header_data)
            current_frames_count = buffer_info['current_frames_count']
            write_index = buffer_info['write_index']

            if current_frames_count == 0:
                return None

            # Calculate frame offset for latest command (same as robot.py)
            latest_index = (write_index - 1) % buffer_size
            frame_offset = frame_offsets[latest_index]

            # Read frame header (timestamp)
            frame_header_data = buf[frame_offset:frame_offset + FRAME_HEADER_SIZE]
            timestamp_ns = int.from_bytes(bytes(frame_header_data), byteorder='little')

            # Read control data
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_end = data_start + frame_data_size
            numpy_dtype = get_dtype(dtype)
            sliced_buf = buf[data_start:data_end]
            data_array = np.frombuffer(bytes(sliced_buf), dtype=numpy_dtype)

            # Reshape if needed (same as robot.py)
            if len(shape) > 0:
                data_array = data_array.reshape(shape)

            return timestamp_ns, data_array.copy()

        except Exception as e:
            self._write_debug_log(f"Error reading control SHM {shm_name}: {e}")
            import traceback
            self._write_debug_log(traceback.format_exc())
            return None

    def _collect_control_data(self) -> None:
        """
        Collect data from all control shared memories.
        Similar to collect_data() but for control SHMs.
        """
        if not self.control_shm_connections:
            print("no control shms to collect data from")
            return

        current_time = time.time()

        for shm_name, conn_info in self.control_shm_connections.items():
            result = self._read_control_shm_data(shm_name)
            if result is None:
                continue

            timestamp_ns, data_array = result

            # Only process if this is new data
            if timestamp_ns != conn_info['last_timestamp']:
                # Queue for async writing
                if self.async_write and self.write_thread_running:
                    try:
                        write_task = {
                            'device_key': shm_name,
                            'device_type': 'ControlSHM',
                            'device_name': shm_name,
                            'timestamp_ns': timestamp_ns,
                            'data_array': data_array.copy(),
                            'device_info': {
                                'type': 'ControlSHM',
                                'name': shm_name,
                                'shape': conn_info['shape'],
                                'dtype': conn_info['dtype']
                            },
                            'collect_time': current_time
                        }
                        self.write_queue.put_nowait(write_task)
                        # logger.info(f"Collected control data from {shm_name}")
                        # logger.info(f"Control data: {data_array}")
                    except queue.Full:
                        try:
                            self.write_queue.put(write_task, timeout=0.01)
                        except queue.Full:
                            self._write_debug_log(f"Write queue full, dropping control data for {shm_name}")

                conn_info['last_timestamp'] = timestamp_ns
                if (
                    self.recording_control is not None
                    and shm_name == self.recording_control.reference_name
                ):
                    self.recording_control.note_reference_commit(timestamp_ns)

    def _handle_recording_control_command(self, command: Dict[str, Any]) -> None:
        """Apply dense recorder commands without changing legacy signals."""
        control = self.recording_control
        if control is None:
            return
        action = command.get('action')
        if action == 'segment_accept':
            control.append_segment('accept', command)
        elif action == 'segment_reject':
            control.append_segment('reject', command)
        elif action == 'shutdown':
            control.mark_incomplete({'code': 'panel_shutdown'})
            self.running = False

    def _disconnect_control_shms(self) -> None:
        """Disconnect from all control shared memories."""
        for shm_name, conn_info in list(self.control_shm_connections.items()):
            try:
                if conn_info['shm']:
                    conn_info['shm'].close()
            except Exception as e:
                self._write_debug_log(f"Error disconnecting from control SHM {shm_name}: {e}")
        self.control_shm_connections.clear()

    def _cleanup_signal_files(self) -> None:
        """Clean up signal files from previous runs."""
        try:
            if self.start_recording_signal.exists():
                self.start_recording_signal.unlink()
            if self.stop_recording_signal.exists():
                self.stop_recording_signal.unlink()
            logger.debug("Cleaned up signal files")
        except Exception as e:
            logger.warning(f"Failed to clean up signal files: {e}")

    def _check_start_signal(self) -> bool:
        """Check if start recording signal exists."""
        if self.start_recording_signal.exists():
            try:
                self.start_recording_signal.unlink()
                self._write_debug_log("Received START recording signal")
                return True
            except Exception as e:
                self._write_debug_log(f"Failed to remove start signal file: {e}")
        return False

    def _check_stop_signal(self) -> bool:
        """Check if stop recording signal exists."""
        if self.stop_recording_signal.exists():
            try:
                self.stop_recording_signal.unlink()
                self._write_debug_log("Received STOP recording signal")
                return True
            except Exception as e:
                self._write_debug_log(f"Failed to remove stop signal file: {e}")
        return False

    def _read_episode_result(self) -> Optional[str]:
        """
        Read and consume the episode result from signal file.

        Returns:
            'success', 'failure', 'discard', or None if no result signal exists
        """
        try:
            if self.episode_result_signal.exists():
                result = self.episode_result_signal.read_text().strip().lower()
                self.episode_result_signal.unlink()
                if result in ('success', 'failure', 'discard'):
                    self._write_debug_log(f"Read episode result: {result}")
                    return result
                else:
                    self._write_debug_log(f"Invalid episode result value: {result}")
        except Exception as e:
            self._write_debug_log(f"Failed to read episode result: {e}")
        return None

    def _save_episode_result_metadata(self, episode_dir: str, result: str) -> None:
        """
        Save episode result metadata to a JSON file in the episode directory.

        Args:
            episode_dir: Episode directory path
            result: 'success' or 'failure'
        """
        try:
            metadata = {
                'episode_result': result,
                'is_success': result == 'success',
                'timestamp': datetime.now().isoformat(),
                'episode_index': self.episode_counts
            }

            metadata_file = Path(episode_dir) / 'episode_result.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            self._write_debug_log(f"Saved episode result metadata: {result} -> {metadata_file}")
            logger.info(f"Episode result saved: {result} (episode {self.episode_counts})")
        except Exception as e:
            self._write_debug_log(f"Failed to save episode result metadata: {e}")
            logger.error(f"Failed to save episode result metadata: {e}")

    def _write_debug_log(self, message: str) -> None:
        """Write debug message to a log file for subprocess debugging."""
        try:
            log_file = Path(self.output_dir) / "inference_saver_debug.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(f"{timestamp} | {message}\n")
        except Exception:
            pass  # Silently fail if we can't write to log

    def _required_episode_topics(self) -> set[str]:
        required = {
            f"nmx/hal/camera/camera_{camera_index}/rgbd"
            for camera_index, _device_key, _device_info in self._camera_entries()
        }
        if any(
            "RizonRobot" in str(device.get("type", ""))
            for device in self.devices.values()
        ):
            for side in ("left", "right"):
                required.add(f"nmx/hal/arm/{side}/tcp/state")
                required.add(f"nmx/hal/hand/{side}/state")
        if any(
            device.get("type") == "YamHeadRobot"
            for device in self.devices.values()
        ):
            required.add("nmx/hal/head/state")
        return required

    @staticmethod
    def _validate_episode_payload(
        episode_dir: str,
        *,
        required_topics: Optional[set[str]] = None,
    ) -> None:
        """Require a nonempty, episode-local NEDF3 chunk index before publish."""
        episode = Path(episode_dir).resolve(strict=True)
        data_dir = episode / "data"
        if data_dir.is_symlink() or not data_dir.is_dir():
            raise RuntimeError("data directory is missing")
        index_path = data_dir / "mcap_index.json"
        if index_path.is_symlink() or not index_path.is_file():
            raise RuntimeError("data/mcap_index.json is missing")
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("data/mcap_index.json is unreadable") from exc
        if not isinstance(index, list) or not index:
            raise RuntimeError("data/mcap_index.json has no shards")

        seen: set[str] = set()
        indexed_topics: set[str] = set()
        for item in index:
            filename = item.get("filename") if isinstance(item, dict) else None
            if (
                not isinstance(filename, str)
                or not filename
                or Path(filename).name != filename
                or filename in seen
            ):
                raise RuntimeError(
                    "data/mcap_index.json contains an invalid shard"
                )
            seen.add(filename)
            topics = item.get("topics")
            if isinstance(topics, list):
                indexed_topics.update(
                    topic for topic in topics if isinstance(topic, str)
                )
            shard = data_dir / filename
            try:
                shard_info = shard.lstat()
                shard.resolve(strict=True).relative_to(
                    data_dir.resolve(strict=True)
                )
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"indexed MCAP shard is unavailable: {filename}"
                ) from exc
            if (
                shard.is_symlink()
                or not shard.is_file()
                or shard_info.st_size <= 0
            ):
                raise RuntimeError(
                    f"indexed MCAP shard is invalid: {filename}"
                )
        missing_topics = set(required_topics or ()) - indexed_topics
        if missing_topics:
            raise RuntimeError(
                "required NEDF3 topics are missing: "
                + ", ".join(sorted(missing_topics))
            )

    def _start_episode(self) -> None:
        """Start a new recording episode."""
        if self.recording_active:
            self._write_debug_log("Recording already active, ignoring start signal")
            return

        try:
            # Create episode directory
            self.episode_dir = os.path.join(self.output_dir, f"episode_{self.episode_counts:04d}_temp")
            self._write_debug_log(f"Opening episode streams: {self.episode_dir}")
            self.open_episode_streams(self.episode_dir)
            if self.recording_control is not None:
                self.recording_control.begin_episode(
                    index=self.episode_counts,
                    episode_dir=Path(self.episode_dir),
                )
            self.recording_active = True
            self._write_debug_log(f"Started recording episode {self.episode_counts}")
        except Exception as e:
            self._write_debug_log(f"Failed to start episode: {e}")
            if self.recording_control is not None:
                self.recording_control.mark_error(
                    {"code": "episode_start_failed", "message": str(e)}
                )
            import traceback
            traceback.print_exc()

    def _stop_episode(self) -> None:
        """Stop current recording episode."""
        if not self.recording_active:
            self._write_debug_log("No active recording, ignoring stop signal")
            return

        try:
            control = self.recording_control
            episode = control.snapshot().get("episode") if control is not None else None
            episode_id = episode.get("id") if isinstance(episode, dict) else None
            if control is not None:
                control.set_state("finalizing", ready=True, active=True)

            # Read episode result before closing (may have been set by PolicyConnector)
            episode_result = self._read_episode_result()
            self.current_episode_result = episode_result

            # Wait for all queued data to be written.
            self._write_debug_log(f"Waiting for async writer to finish... queue_size={self.write_queue.qsize()}")
            if not self._wait_for_write_queue():
                self._record_episode_write_failure(
                    "NEDF3 async writer did not drain cleanly"
                )

            # Closing the SDK writer flushes the final shard and index.
            required_topics = self._required_episode_topics()
            self.close_episode_streams()

            payload_error = self._episode_write_failure_reason
            if episode_result in ('success', 'failure') and self.episode_dir:
                if payload_error is None:
                    try:
                        self._validate_episode_payload(
                            self.episode_dir,
                            required_topics=required_topics,
                        )
                    except (OSError, RuntimeError) as exc:
                        payload_error = (
                            f"NEDF3 episode payload is incomplete: {exc}"
                        )
                        self._record_episode_write_failure(payload_error)
                if payload_error is not None:
                    self._write_debug_log(payload_error)
                    self._mark_episode_incomplete(payload_error)
                    if control is not None:
                        control.mark_incomplete(
                            {
                                "code": "episode_payload_incomplete",
                                "message": payload_error,
                            }
                        )
                    self.recording_active = False
                    self.current_episode_result = None
                    self.running = False
                    return

            self._write_debug_log("All data written to NEDF3")

            # Save episode result metadata before finalizing (while we still have temp dir)
            if episode_result in ('success', 'failure') and self.episode_dir:
                self._save_episode_result_metadata(self.episode_dir, episode_result)
                self._write_debug_log(f"Finalizing NEDF3 episode: {self.episode_dir}")
                self.last_finalized_episode_dir = None
                self._finalize_nedf3_episode(self.episode_dir)
                if control is not None:
                    final_dir = self.last_finalized_episode_dir
                    if not final_dir or not Path(final_dir).is_dir():
                        raise RuntimeError(
                            "NEDF3 episode finalization did not produce a final directory"
                        )
                    if episode_id:
                        write_conversion_source_receipt(
                            Path(final_dir),
                            episode_id=episode_id,
                            episode_index=self.episode_counts - 1,
                            result=episode_result,
                            control=control,
                            config_name=os.environ.get("SUPERINFERENCE_CONFIG_NAME"),
                        )
                    control.arm()
            elif episode_result == 'discard' and self.episode_dir:
                self._write_debug_log(f"Discarding NEDF3 episode: {self.episode_dir}")
                self._discard_nedf3_episode(self.episode_dir)
                if control is not None:
                    control.arm()
            else:
                self._write_debug_log(f"Invalid episode result: {episode_result}, or episode_dir is none, not saving episode data")
                if self.episode_dir:
                    self._discard_nedf3_episode(self.episode_dir)
                if control is not None and episode_result not in ('success', 'failure'):
                    control.mark_incomplete({"code": "missing_episode_result"})

            self.recording_active = False
            self.current_episode_result = None
            self._write_debug_log(f"Stopped recording episode {self.episode_counts - 1}")
        except Exception as e:
            self._write_debug_log(f"Failed to stop episode: {e}")
            if self.recording_control is not None:
                self.recording_control.mark_error(
                    {"code": "episode_stop_failed", "message": str(e)}
                )
            import traceback
            traceback.print_exc()

    def run_inference_saver(self) -> None:
        """Run the inference data saver - controlled via signal files."""
        if not self.summary_shm:
            logger.error("Not connected to summary SHM")
            if self.recording_control is not None:
                self.recording_control.mark_error(
                    {"code": "summary_shm_unavailable", "message": "Not connected to summary SHM"}
                )
            return

        self.running = True

        # Connect to control SHMs if configured
        if self.control_shm_configs:
            self._connect_control_shms()
            logger.info(f"Connected to {len(self.control_shm_connections)} control SHM(s)")

        # Start async writer
        if self.async_write:
            self.start_async_writer()

        if self.recording_control is not None:
            self.recording_control.arm()
            logger.info("Async write mode enabled - data will be written continuously to disk")
        else:
            logger.warning("Batch mode not supported for inference saver, enabling async mode")
            self.async_write = True
            self.start_async_writer()

        # Send ready signal to parent process
        if self.ready_conn:
            try:
                self.ready_conn.send({'status': 'ready'})
                logger.info("Sent ready signal to parent process")
            except Exception as e:
                logger.warning(f"Failed to send ready signal: {e}")

        logger.info("Inference data saver started.")
        if self.auto_start:
            logger.info("Auto-start mode enabled - recording will begin immediately")
        else:
            logger.info("Waiting for recording signals...")
            logger.info("Create signal files to control recording:")
            logger.info(f"  Start: {self.start_recording_signal}")
            logger.info(f"  Stop: {self.stop_recording_signal}")

        # Create precise timing function for collection loop
        collection_interval = 0.001
        wait_for_next_collect = precise_loop_timing(collection_interval)

        # Auto-start recording if configured
        if self.auto_start:
            self._start_episode()

        frame_count = 0
        last_log_time = time.time()
        self._write_debug_log(f"Starting main loop, auto_start={self.auto_start}, recording_active={self.recording_active}")
        try:
            while self.running:
                if self.recording_control is not None:
                    self.recording_control.process_commands(
                        self._handle_recording_control_command,
                        allowed_actions={'segment_accept', 'segment_reject', 'shutdown'},
                    )

                # Check for start/stop signals
                if self._check_start_signal() and not self.recording_active:
                    self._write_debug_log("Starting episode from signal")
                    self._start_episode()

                if self._check_stop_signal() and self.recording_active:
                    self._write_debug_log("Stopping episode from signal")
                    self._stop_episode()

                # Collect data if recording is active
                if self.recording_active:
                    self.collect_data()
                    # Also collect control SHM data
                    self._collect_control_data()
                    frame_count += 1

                    # Log progress every 5 seconds
                    current_time = time.time()
                    if current_time - last_log_time >= 5.0:
                        self._write_debug_log(f"Recording in progress - {frame_count} collection cycles, queue_size={self.write_queue.qsize()}")
                        last_log_time = current_time

                # Wait for next collection interval
                wait_for_next_collect()

        except KeyboardInterrupt:
            self._write_debug_log("Interrupted by KeyboardInterrupt (SIGINT)")
        finally:
            self._write_debug_log(f"Entering finally block, recording_active={self.recording_active}, frame_count={frame_count}")
            # Stop any active recording - this ensures data is saved on Ctrl+C
            if self.recording_active:
                self._write_debug_log("Saving active recording before exit...")
                self._stop_episode()

            self._write_debug_log("Calling stop()")
            self.stop()

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """
        Override signal handler to ensure data is saved before stopping.

        The base class signal handler calls stop() directly, which would
        discard queued data. We need to stop the recording first.
        """
        self._write_debug_log(f"Signal handler received signal {signum}")
        logger.info(f"Received signal {signum}, shutting down...")

        # First, stop recording to flush all queued data
        if self.recording_active:
            self._write_debug_log("Signal handler: Stopping active recording first")
            self._stop_episode()

        # Now stop the consumer
        self.stop()
        sys.exit(0)

    def stop(self) -> None:
        """Stop the data saver consumer and async writer."""
        if not self.running:
            return

        logger.info("Stopping inference data saver consumer...")

        # Stop async writer first
        self.stop_async_writer()

        # Clean up signal files
        self._cleanup_signal_files()

        # Disconnect from control SHMs
        self._disconnect_control_shms()

        # Call parent stop method (but skip its stop_async_writer call)
        self.running = False
        if self.summary_shm:
            try:
                self.summary_shm.close()
            except Exception:
                pass
            self.summary_shm = None

        logger.info("Inference data saver consumer stopped")


def _get_next_episode_index(output_dir: str, default_start: int = 0) -> int:
    """
    Get the next episode index by scanning existing episode directories.

    Args:
        output_dir: Output directory containing episode folders
        default_start: Default start index if no episodes exist

    Returns:
        Next episode index (max existing + 1, or default_start if none exist)
    """
    import re

    output_path = Path(output_dir)
    if not output_path.exists():
        return default_start

    max_index = -1
    episode_pattern = re.compile(r'^episode_(\d+)(?:_temp)?$')

    for item in output_path.iterdir():
        if item.is_dir():
            match = episode_pattern.match(item.name)
            if match:
                episode_idx = int(match.group(1))
                max_index = max(max_index, episode_idx)

    if max_index >= 0:
        return max_index + 1
    return default_start


def _copy_config_to_output(config_file: str, output_dir: str) -> None:
    """
    Copy the config file to the output directory.

    Args:
        config_file: Path to the config file
        output_dir: Output directory to copy to
    """
    import shutil

    config_path = Path(config_file)
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_file}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy config file with original name
    dest_path = output_path / config_path.name

    # If file already exists, check if it's the same
    if dest_path.exists():
        # Compare contents
        try:
            with open(config_path, 'r') as f1, open(dest_path, 'r') as f2:
                if f1.read() == f2.read():
                    logger.info(f"Config file already exists and is identical: {dest_path}")
                    return
        except Exception:
            pass

        # Different content, create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{config_path.stem}_{timestamp}{config_path.suffix}"
        dest_path = output_path / backup_name
        logger.info(f"Config file exists with different content, saving as: {backup_name}")

    try:
        shutil.copy2(config_path, dest_path)
        logger.info(f"Copied config file to: {dest_path}")
    except Exception as e:
        logger.error(f"Failed to copy config file: {e}")


def main() -> None:
    """Main function to run the inference data saver consumer."""
    parser = argparse.ArgumentParser(description="Inference Data Saver Consumer - Save Device Data during Inference to NEDF3 Format")
    parser.add_argument("--summary-shm", "-s", default="device_summary_data",
                        help="Summary shared memory name (default: device_summary_data)")
    parser.add_argument("--output-dir", "-o", default="inference_rollout_data",
                        help="Output directory for saved data (default: inference_rollout_data)")
    parser.add_argument("--interval", "-i", type=float, default=0.001,
                        help="Save interval in seconds (default: 0.001)")
    parser.add_argument("--start-idx", "-st", type=int, default=0,
                        help="Start index of the episode")
    parser.add_argument("--image-size", "-is", type=str, default=None,
                        help="Target image size for camera data as 'width,height' (e.g., '400,300'). "
                             "If not specified, save at original size.")
    parser.add_argument("--queue-size", "-q", type=int, default=5000,
                        help="Async write queue size (default: 5000)")
    parser.add_argument("--robot-type", "-r", type=str, default="Rizon",
                        help="Robot type for NEDF metadata (default: Rizon)")
    parser.add_argument("--task-name", "-t", type=str, default="inference_rollout",
                        help="Task name for NEDF metadata (default: inference_rollout)")
    parser.add_argument("--auto-start", "-a", action="store_true", default=False,
                        help="Auto-start recording immediately (default: False)")
    parser.add_argument("--ready-file", type=str, default=None,
                        help="File path to write 'ready' when saver is initialized (for subprocess coordination)")
    parser.add_argument("--control-shms", "-c", type=str, default=None,
                        help="JSON string of control SHM configurations, e.g., "
                             "'[{\"shm\": \"RizonRobot_1_control\", \"shape\": [16], \"dtype\": \"float64\"}]'")
    parser.add_argument("--config-file", type=str, default=None,
                        help="Path to the config file to copy to output directory")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing episodes (start from start_idx). "
                             "If not set, auto-detect next episode index from existing episodes.")
    parser.add_argument("--anomaly-config", type=str, default=None,
                        help="Path to anomaly detection config file")
    parser.add_argument("--signal-files", type=json.loads, default=None,
                        help="Dictionary of signal file paths")
    args = parser.parse_args()

    # Parse image size
    image_size = None
    if args.image_size:
        try:
            width, height = map(int, args.image_size.split(','))
            image_size = (width, height)
            logger.info(f"Target image size: {image_size} (width, height)")
        except ValueError:
            logger.error(f"Invalid image size format: {args.image_size}. Expected 'width,height' (e.g., '400,300')")
            return

    # Parse control SHMs
    control_shms = None
    if args.control_shms:
        try:
            control_shms = json.loads(args.control_shms)
            logger.info(f"Control SHMs to record: {control_shms}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid control-shms JSON format: {e}")
            return

    # Determine start index
    start_idx = args.start_idx
    if not args.overwrite:
        # Auto-detect next episode index from existing episodes
        start_idx = _get_next_episode_index(args.output_dir, args.start_idx)
        if start_idx != args.start_idx:
            logger.info(f"Auto-detected next episode index: {start_idx} (use --overwrite to start from {args.start_idx})")

    # Copy config file to output directory if specified
    if args.config_file:
        logger.info(f"Config file specified: {args.config_file}")
        _copy_config_to_output(args.config_file, args.output_dir)
    else:
        logger.info("No config file specified via --config-file argument")

    # Create inference data saver consumer
    consumer = InferenceDataSaverConsumer(
        summary_shm_name=args.summary_shm,
        output_dir=args.output_dir,
        save_interval=args.interval,
        start_idx=start_idx,
        image_size=image_size,
        async_write=True,
        queue_size=args.queue_size,
        robot_type=args.robot_type,
        task_name=args.task_name,
        auto_start=args.auto_start,
        control_shms=control_shms,
        anomaly_config_path=args.anomaly_config,
        signal_files=args.signal_files
    )

    logger.info("Inference Data Saver Consumer - Save Device Data during Inference to NEDF3 Format")
    logger.info("=" * 80)
    logger.info(f"Summary SHM: {args.summary_shm}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Save interval: {args.interval}s")
    logger.info(f"Start episode index: {start_idx}")
    logger.info(f"Robot type: {args.robot_type}")
    logger.info(f"Task name: {args.task_name}")
    logger.info(f"Queue size: {args.queue_size}")
    logger.info(f"Auto-start: {args.auto_start}")
    logger.info(f"Overwrite mode: {args.overwrite}")
    if image_size:
        logger.info(f"Image resize: {image_size[0]}x{image_size[1]}")
    if control_shms:
        logger.info(f"Control SHMs: {[c['shm'] for c in control_shms]}")
    if args.config_file:
        logger.info(f"Config file: {args.config_file}")
    logger.info("")

    # Try to connect
    if not consumer.connect():
        logger.error("Failed to connect to summary SHM. Make sure the Device Manager is running.")
        return

    # Write ready signal to file if specified (for subprocess coordination)
    if args.ready_file:
        try:
            ready_path = Path(args.ready_file)
            ready_path.parent.mkdir(parents=True, exist_ok=True)
            ready_path.write_text("ready")
            logger.info(f"Wrote ready signal to {args.ready_file}")
        except Exception as e:
            logger.warning(f"Failed to write ready file: {e}")

    try:
        consumer.run_inference_saver()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        consumer.stop()


if __name__ == "__main__":
    main()
