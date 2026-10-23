#!/usr/bin/env python3
"""
Asynchronous Data Saver Consumer - Save device manager data to local storage.

This module provides data saving functionality by consuming device manager data
and storing it to various local storage formats.

Author: Zheng Wang
"""

import imageio

import sys
import time
import signal
import argparse
import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import queue
import threading

# Ensure project root is on PYTHONPATH when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger_config import logger
from utils.time_control import precise_loop_timing
try:
    from .data_saver import DataSaverConsumer as SyncDataSaverConsumer
    from .data_saver import wait_enter_blocking, enter_pressed_nonblocking
except ImportError:
    from data_saver import DataSaverConsumer as SyncDataSaverConsumer
    from data_saver import wait_enter_blocking, enter_pressed_nonblocking

class DataSaverConsumer(SyncDataSaverConsumer):
    """
    Data saver consumer for storing device manager data.

    This consumer saves data based on device type:
    - Camera devices: Save as images (PNG/JPG)
    - Base devices: Save as NPZ files
    - Uses frame timestamp as folder name
    """

    def __init__(self, summary_shm_name: str = "device_summary_data",
                 output_dir: str = "saved_data", save_interval: float = 0.04,
                 start_idx:int=0, image_size: Optional[Tuple[int, int]] = None, async_write: bool = True, queue_size: int = 100000) -> None:
        """
        Initialize the data saver consumer.

        Args:
            summary_shm_name: Name of the summary shared memory
            output_dir: Directory to save data files
            save_interval: Interval between saves in seconds
            async_write: Whether to use async writing thread (default: True)
            queue_size: Size of async write queue (default: 100000)
        """
        super().__init__(summary_shm_name, output_dir, save_interval, start_idx, image_size)

        # Async writing infrastructure
        self.async_write = async_write
        self.write_queue = queue.Queue(maxsize=queue_size)
        self.write_thread = None
        self.write_thread_running = False

        # File stream management for async mode
        self.device_file_streams: Dict[str, Any] = {}  # device_key -> file handle
        self.device_data_counts: Dict[str, int] = {}   # device_key -> data count
        self.current_episode_dir = None
        self.episode_active = False

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def open_episode_streams(self, episode_dir: str) -> None:
        """
        Open file streams for each device for the current episode.

        Args:
            episode_dir: Directory path for this episode
        """
        try:
            os.makedirs(episode_dir, exist_ok=True)
            self.current_episode_dir = episode_dir

            # Open a pickle file stream for each device
            for device_key in self.devices:
                device_info = self.devices[device_key]
                device_type = device_info['type']

                # Create filename based on device
                if device_type in ['OpenCVCameraDevice','HikCameraDevice']:
                    # For cameras, use pickle to store list of (timestamp, image) pairs
                    filename = f"{device_key}.pkl"
                else:
                    # For other devices, use pickle
                    filename = f"{device_key}.pkl"

                filepath = os.path.join(episode_dir, filename)

                # Open file in binary append mode
                self.device_file_streams[device_key] = open(filepath, 'wb')
                self.device_data_counts[device_key] = 0

                logger.debug(f"Opened stream for {device_key}: {filepath}")

            self.episode_active = True
            logger.info(f"Opened {len(self.device_file_streams)} device streams in {episode_dir}")

        except Exception as e:
            logger.error(f"Error opening episode streams: {e}")
            raise

    def close_episode_streams(self) -> None:
        """Close all device file streams."""
        try:
            for device_key, file_handle in self.device_file_streams.items():
                if file_handle and not file_handle.closed:
                    file_handle.close()
                    logger.debug(f"Closed stream for {device_key}, wrote {self.device_data_counts[device_key]} frames")

            logger.info("Closed all device streams")
            self.device_file_streams.clear()
            self.device_data_counts.clear()
            self.episode_active = False

        except Exception as e:
            logger.error(f"Error closing episode streams: {e}")

    def _async_write_worker(self) -> None:
        """
        Background worker thread that continuously consumes data from write queue.

        This thread runs independently, writing data to disk as it becomes available.
        Uses file streams - each device has its own pickle file that is appended to.
        """
        logger.info("Async write worker thread started")

        while self.write_thread_running:
            try:
                # Block for up to 0.1s waiting for data
                write_task = self.write_queue.get(timeout=0.1)

                # Extract task data
                device_key = write_task['device_key']
                device_type = write_task['device_type']
                device_name = write_task['device_name']
                timestamp_ns = write_task['timestamp_ns']
                data_array = write_task['data_array']
                device_info = write_task['device_info']

                # Check if episode is active and stream is open
                if not self.episode_active or device_key not in self.device_file_streams:
                    logger.warning(f"Episode not active or stream not open for {device_key}, dropping data")
                    self.write_queue.task_done()
                    continue

                # Get file handle for this device
                file_handle = self.device_file_streams[device_key]

                # Resize camera images if needed
                if device_type in ['OpenCVCameraDevice', 'HikCameraDevice'] and self.image_size is not None:
                    import cv2
                    target_width, target_height = self.image_size
                    original_shape = data_array.shape
                    data_array = cv2.resize(data_array, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                    logger.debug(f"Resized image from {original_shape[:2]} to {data_array.shape[:2]}")

                # Prepare data record: (timestamp_ns, data_array)
                data_record = {
                    'timestamp_ns': timestamp_ns,
                    'data': data_array,
                    'device_info': {
                        'device_type': device_info['type'],
                        'device_class_id': device_info['id'],
                        'fps': device_info['fps'],
                        'hardware_latency_ms': device_info['hardware_latency_ms']
                    }
                }

                # Write to pickle file stream
                pickle.dump(data_record, file_handle)
                file_handle.flush()  # Ensure data is written to disk

                # Update counter
                self.device_data_counts[device_key] += 1

                # Log progress periodically
                if self.device_data_counts[device_key] % 100 == 0:
                    logger.debug(f"{device_key}: wrote {self.device_data_counts[device_key]} frames")

                self.write_queue.task_done()

            except queue.Empty:
                # No data available, continue waiting
                continue
            except Exception as e:
                logger.error(f"Error in async write worker: {e}")
                import traceback
                traceback.print_exc()

        logger.info("Async write worker thread stopped")

    def start_async_writer(self) -> None:
        """Start the background async writer thread."""
        if self.async_write and not self.write_thread_running:
            self.write_thread_running = True
            self.write_thread = threading.Thread(target=self._async_write_worker, daemon=True)
            self.write_thread.start()
            logger.info("Started async write thread")

    def stop_async_writer(self) -> None:
        """Stop the background async writer thread and close file streams."""
        if self.write_thread_running:
            logger.info("Stopping async write thread...")
            self.write_thread_running = False

            if self.write_thread:
                self.write_thread.join(timeout=5.0)
                if self.write_thread.is_alive():
                    logger.warning("Async write thread did not stop gracefully")
                else:
                    logger.info("Async write thread stopped")

            # Process any remaining items in queue
            remaining = 0
            while not self.write_queue.empty():
                try:
                    self.write_queue.get_nowait()
                    remaining += 1
                except queue.Empty:
                    break

            if remaining > 0:
                logger.warning(f"Discarded {remaining} items from write queue")

            # Close any open file streams
            if self.episode_active:
                self.close_episode_streams()

    def collect_data(self) -> None:
        """
        Collect data from all devices in a non-blocking manner.

        Each device is polled independently:
        - High-frequency devices (e.g., cameras) update frequently
        - Low-frequency devices (e.g., robot states) update less often
        - Only new data (based on timestamp changes) is collected
        - Data is either queued for async writing or buffered for batch saving
        """
        current_time = time.time()

        # Read each device independently - non-blocking
        # This allows high-freq devices to update without waiting for low-freq ones
        for device_key in self.devices:
            # Read latest data for this device from shared memory
            # _read_device_data() is non-blocking - just reads from SHM buffer
            data = self._read_device_data(device_key)

            if data is not None:
                timestamp_ns, data_array = data

                # Only process if this is new data (timestamp changed)
                if timestamp_ns != self.devices[device_key]['last_timestamp']:

                    # Option 1: Async writing - queue data for background thread
                    if self.async_write and self.write_thread_running:
                        try:
                            # Non-blocking queue put - discard if queue full
                            device_info = self.devices[device_key]
                            write_task = {
                                'device_key': device_key,
                                'device_type': device_info['type'],
                                'device_name': device_info['name'],
                                'timestamp_ns': timestamp_ns,
                                'data_array': data_array.copy(),  # Copy to avoid race conditions
                                'device_info': device_info,
                                'collect_time': current_time
                            }
                            self.write_queue.put_nowait(write_task)
                        except queue.Full:
                            logger.warning(f"Write queue full, dropping data for {device_key}")

                    # Option 2: Buffer data for batch saving (original behavior)
                    else:
                        self.data_buffer[device_key].append((timestamp_ns, data_array))

                    # Update tracking
                    self.devices[device_key]['last_timestamp'] = timestamp_ns
                    self.devices[device_key]['last_update_time'] = current_time

        # Note: We do NOT use read_all_device_data() here because:
        # 1. It reads ALL devices from the SAME frame timestamp
        # 2. This would force all devices to be synchronized
        # 3. We want each device to update independently at its own frequency

    def _convert_pickle_streams_to_hdf5(self, episode_dir: str, remove=True) -> None:
        """
        Convert pickle stream files to a single HDF5 episode file.

        Saves both data and timestamps for each device to enable sensor synchronization.

        Args:
            episode_dir: Directory containing pickle files for each device
            remove: Whether to remove pickle files after conversion
        """
        try:
            import h5py
            import shutil
            import traceback

            episode_data = {}
            episode_timestamps = {}
            episode_path = os.path.join(self.output_dir, f"episode_{self.episode_counts:04d}.hdf5")

            logger.info(f"Converting pickle streams to HDF5: {episode_path}")

            # Read data from each device's pickle file
            for device_key in self.devices:
                pickle_file = os.path.join(episode_dir, f"{device_key}.pkl")

                if not os.path.exists(pickle_file):
                    logger.warning(f"Pickle file not found: {pickle_file}")
                    continue

                device_data_list = []
                device_timestamp_list = []

                # Read all pickled records from file
                with open(pickle_file, 'rb') as f:
                    try:
                        while True:
                            data_record = pickle.load(f)
                            device_data_list.append(data_record['data'])
                            device_timestamp_list.append(data_record['timestamp_ns'])
                    except EOFError:
                        # End of file reached
                        pass

                if device_data_list:
                    episode_data[device_key] = device_data_list
                    episode_timestamps[device_key] = device_timestamp_list
                    logger.info(f"Loaded {len(device_data_list)} frames from {device_key}")

            if not episode_data:
                logger.warning("No data to save in episode")
                return

            # Find the earliest timestamp across all devices to use as t0
            first_timestamps = [ts[0] for ts in episode_timestamps.values() if ts]
            if not first_timestamps:
                global_start_time_ns = 0
                logger.warning("No timestamps found, cannot normalize.")
            else:
                global_start_time_ns = min(first_timestamps)

            logger.info(f"Normalizing timestamps with t0={global_start_time_ns} ns")

            # Normalize and convert timestamps to seconds
            for device_key, ts_list in episode_timestamps.items():
                if not ts_list:
                    continue
                timestamps_ns = np.array(ts_list, dtype=np.float64)
                timestamps_sec = (timestamps_ns - global_start_time_ns) / 1e9
                episode_timestamps[device_key] = timestamps_sec

            # Save to HDF5 format
            with h5py.File(episode_path, 'w') as root:
                # Store the original start time as a root attribute for reference
                root.attrs['start_time_ns'] = global_start_time_ns

                root.create_dataset('freq', data=int(1.0/self.save_interval))
                root.create_dataset('timestamp', data=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                      dtype=h5py.string_dtype(encoding='utf-8'))

                # Save data and timestamps for each device in its own group
                for device_key, val_list in episode_data.items():
                    # Create a group for the device
                    device_group = root.create_group(device_key)

                    # Save device data
                    if isinstance(val_list[0], np.ndarray):
                        device_group.create_dataset('data', data=np.stack(val_list))
                    elif isinstance(val_list[0], str):
                        device_group.create_dataset('data', data=val_list, dtype=h5py.string_dtype(encoding='utf-8'))
                    else:
                        raise ValueError(f"Failed to save data for {device_key}:{type(val_list[0])}")

                    # Save corresponding timestamps (in seconds, normalized)
                    device_group.create_dataset('timestamps', data=episode_timestamps[device_key])

                    logger.debug(f"Saved group {device_key}: {len(val_list)} frames with timestamps")

                logger.info(f"Created episode HDF5: {episode_path} with normalized timestamps in seconds.")
                self.episode_counts += 1
        except Exception as e:
            if os.path.exists(episode_path):
                os.remove(episode_path)
            traceback.print_exc()
            logger.error(f"Error converting pickle streams to HDF5: {e}")

        finally:
            # Remove pickle files and episode directory if requested
            if remove and os.path.exists(episode_dir):
                try:
                    shutil.rmtree(episode_dir)
                    logger.info(f"Removed temporary episode directory: {episode_dir}")
                except Exception as e:
                    logger.error(f"Error removing episode directory {episode_dir}: {e}")

    def _convert_data_to_episode(self, data_folders: List[str], remove=True) -> None:
        """Organize the saved data folders into one episode .h5 file"""
        assert len(data_folders)==len(set(data_folders)), "Data folders have redundancy"
        if len(data_folders)==0: return
        try:
            import cv2
            import h5py
            import shutil
            import traceback
            episode_data = {}
            episode_path = os.path.join(self.output_dir, f"episode_{self.episode_counts:04d}.hdf5")

            ALL_IMAGE_SUFFIX = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
            for frame_i in data_folders:
                # Load data from each frame folder
                for dfile_j in os.listdir(frame_i):
                    if dfile_j.endswith('.npz'):
                        data_dict = np.load(os.path.join(frame_i, dfile_j), allow_pickle=True)
                        meta_data = eval(data_dict['metadata'][0])
                        # Generate key from meta_data
                        dkey = "_".join([
                            meta_data.get('device_type', 'Unknown'),
                            str(meta_data.get('device_class_id', 0))
                        ])
                        content = data_dict.get('data', [])

                    elif any(dfile_j.endswith(img_suffix) for img_suffix in ALL_IMAGE_SUFFIX):
                        # process image observation
                        dkey = "_".join(dfile_j.split('_')[:2])
                        # (H, W, C) with RGB format
                        content =  cv2.cvtColor(cv2.imread(os.path.join(frame_i, dfile_j)), cv2.COLOR_BGR2RGB)
                    else:
                        logger.info(f"Failed to recognize file {dfile_j}:{type(dfile_j)}")
                        dkey = content = None
                    if dkey is not None:
                        # Update episode data
                        if dkey not in episode_data: episode_data[dkey] = [content]
                        else: episode_data[dkey].append(content)
            # Save into .hdf5 format
            with h5py.File(episode_path, 'w') as root:
                root.create_dataset('freq', data=int(1.0/self.save_interval))
                root.create_dataset('timestamp', data=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), dtype=h5py.string_dtype(encoding='utf-8'))
                for key, val in episode_data.items():
                    if isinstance(val[0], np.ndarray):
                        root.create_dataset(key, data=np.stack(val))
                    elif isinstance(val[0], str):
                        root.create_dataset(key, data=val, dtype=h5py.string_dtype(encoding='utf-8'))
                    else:
                        raise ValueError(f"Failed to save data for {key}:{type(val[0])}")

            # Update episode id
            self.episode_counts += 1
        except Exception as e:
            if os.path.exists(episode_path): os.remove(episode_path)
            traceback.print_exc()
            logger.error(f"Error when organizing data to the {self.episode_counts:04d}th episode")

        finally:
            # Remove original data folder after saving them
            if remove:
                for data_folder in data_folders:
                    try:
                        shutil.rmtree(data_folder)
                    except FileNotFoundError as e:
                        logger.error(f"Error when removing data folder {data_folder}: {e}")

    def start_saving(self) -> None:
        """Start the data saving process."""
        if not self.summary_shm:
            logger.error("Not connected to summary SHM")
            return

        self.running = True

        # Start async writer if enabled
        if self.async_write:
            self.start_async_writer()
            logger.info("Async write mode enabled - data will be written continuously to disk")
        else:
            logger.info("Batch mode enabled - data will be saved at intervals")

        logger.info("Start data acquisition. Ctrl+C to exit.")
        logger.info("+"*40)

        try:
            while self.running:
                logger.info("Press Enter to start data acquisition.")
                wait_enter_blocking()
                data_folders = []
                logger.info("Data acquisition started. Press Enter to stop.")
                time_episode_start = time.time()

                # Async mode: Open file streams for this episode
                if self.async_write:
                    episode_dir = os.path.join(self.output_dir, f"episode_{self.episode_counts:04d}_temp")
                    self.open_episode_streams(episode_dir)

                # Create precise timing function for collection loop
                # In async mode, this is just the polling rate
                # In batch mode, this is the save interval
                collection_interval = 0.005 if self.async_write else self.save_interval
                wait_for_next_collect = precise_loop_timing(collection_interval)

                while True:
                    # Collect data from devices - non-blocking, independent per device
                    self.collect_data()

                    # Batch mode: save data at intervals
                    if not self.async_write:
                        if (self.frame_timestamp is not None and
                            self.last_saving_timestamp != self.frame_timestamp and
                            any(self.data_buffer.values())):

                            saved_data_folder = self.save_data()
                            if saved_data_folder is not None:
                                data_folders.append(saved_data_folder)
                                self.last_saving_timestamp = self.frame_timestamp

                            if len(data_folders) != len(set(data_folders)):
                                logger.error("Data folders have redundancy")

                    # Wait for next collection interval
                    wait_for_next_collect()

                    if enter_pressed_nonblocking():
                        logger.info("Data acquisition stopped by user.")
                        break

                # Batch mode: save any remaining data
                if not self.async_write:
                    if all(self.data_buffer.values()) and self.last_saving_timestamp != self.frame_timestamp:
                        saved_data_folder = self.save_data()
                        if saved_data_folder is not None and saved_data_folder not in data_folders:
                            data_folders.append(saved_data_folder)

                time_episode_end = time.time()

                # Async mode: Close streams and convert to HDF5
                if self.async_write:
                    logger.info("Waiting for async writer to finish...")
                    self.write_queue.join()  # Wait for all queued data to be written
                    logger.info("All data written to disk")

                    # Close file streams
                    self.close_episode_streams()

                    # Convert pickle streams to HDF5
                    logger.info("Converting pickle streams to HDF5...")
                    self._convert_pickle_streams_to_hdf5(episode_dir, remove=True)
                else:
                    # Batch mode: organize data folders into episode
                    logger.info(f"Time taken to collect data: {time_episode_end - time_episode_start}s")
                    if data_folders:
                        logger.info(f"Actual frequency: {len(data_folders)/(time_episode_end - time_episode_start):.2f}Hz")

                    logger.info("Organizing Data into Episode...")
                    if len(data_folders) > 0:
                        self._convert_data_to_episode(data_folders)

                logger.info(f"Episode completed in {time.time() - time_episode_start:.2f}s")

        except KeyboardInterrupt:
            logger.info("Data saving interrupted by user")

        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the data saver consumer and async writer."""
        if not self.running:
            return

        logger.info("Stopping data saver consumer...")

        # Stop async writer first
        self.stop_async_writer()

        # Call parent stop method
        super().stop()

        logger.info("Data saver consumer stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get status of the data saver consumer."""
        status = {
            'connected': self.summary_shm is not None,
            'running': self.running,
            'devices': len(self.devices),
            'output_dir': self.output_dir,
            'save_interval': self.save_interval,
            'frame_timestamp': self.frame_timestamp,
            'async_write': self.async_write,
            'write_thread_running': self.write_thread_running,
            'queue_size': self.write_queue.qsize() if self.async_write else 0
        }

        # Add buffer information
        buffer_info = {}
        for device_key, buffer_data in self.data_buffer.items():
            buffer_info[device_key] = len(buffer_data)
        status['buffer_samples'] = buffer_info

        return status


def main() -> None:
    """Main function to run the data saver consumer."""
    parser = argparse.ArgumentParser(description="Data Saver Consumer - Save Device Data")
    parser.add_argument("--summary-shm", "-s", default="device_summary_data",
                        help="Summary shared memory name (default: device_summary_data)")
    parser.add_argument("--output-dir", "-o", default="saved_data",
                        help="Output directory for saved data (default: saved_data)")
    parser.add_argument("--interval", "-i", type=float, default=0.001,
                        help="Save interval in seconds (default: 1.0)")
    parser.add_argument("--start-idx", "-st", type=int, default=0,
                        help="Start index of the episode")
    parser.add_argument("--image-size", "-is", type=str, default=None,
                        help="Target image size for camera data as 'width,height' (e.g., '400,300'). "
                             "If not specified, save at original size.")
    parser.add_argument("--async-write", "-a", action="store_true", default=True,
                        help="Use async write mode (default: True)")
    parser.add_argument("--batch-mode", "-b", action="store_true",
                        help="Use batch mode instead of async write")
    parser.add_argument("--queue-size", "-q", type=int, default=1000,
                        help="Async write queue size (default: 1000)")
    parser.add_argument("--status", action="store_true",
                        help="Show consumer status and exit")

    args = parser.parse_args()

    # Determine write mode
    async_write = not args.batch_mode if args.batch_mode else args.async_write

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

    # Create data saver consumer
    consumer = DataSaverConsumer(
        summary_shm_name=args.summary_shm,
        output_dir=args.output_dir,
        save_interval=args.interval,
        start_idx=args.start_idx,
        image_size=image_size,
        async_write=async_write,
        queue_size=args.queue_size,
    )

    if args.status:
        # Show status
        status = consumer.get_status()
        logger.info("Data Saver Consumer Status:")
        logger.info(f"Connected: {status['connected']}")
        logger.info(f"Running: {status['running']}")
        logger.info(f"Devices: {status['devices']}")
        logger.info(f"Output directory: {status['output_dir']}")
        logger.info(f"Save interval: {status['save_interval']}s")
        logger.info(f"Async write mode: {status['async_write']}")
        logger.info(f"Write thread running: {status['write_thread_running']}")
        logger.info(f"Write queue size: {status['queue_size']}")
        logger.info(f"Frame timestamp: {status['frame_timestamp']}")

        if status['buffer_samples']:
            logger.info("\nBuffer samples:")
        for device_key, samples in status['buffer_samples'].items():
            logger.info(f"  {device_key}: {samples} samples")
        return

    logger.info("Data Saver Consumer - Save Device Data")
    logger.info("======================================")
    logger.info(f"Summary SHM: {args.summary_shm}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Save interval: {args.interval}s")
    logger.info(f"Write mode: {'Async (non-blocking)' if async_write else 'Batch (synchronized)'}")
    if async_write:
        logger.info(f"Queue size: {args.queue_size}")
    logger.info("")

    # Try to connect
    if not consumer.connect():
        logger.error("Failed to connect to summary SHM. Make sure the Device Manager is running.")
        return

    try:
        # consumer.start_saving()
        consumer.start_saving()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        consumer.stop()


if __name__ == "__main__":
    main()
