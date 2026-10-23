#!/usr/bin/env python3
"""
Data Saver Consumer - Save device manager data to local storage.

This module provides data saving functionality by consuming device manager data
and storing it to various local storage formats.

Author: Jun Lv, Zheng Wang
"""

import imageio
import cv2
import h5py
import shutil
import traceback
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

from consumer.core.base import BaseConsumer

from utils.logger_config import logger
from utils.time_control import precise_loop_timing

try:
    from pynput import keyboard as pynput_keyboard
except (ImportError, OSError):
    pynput_keyboard = None

if sys.platform == 'win32':
    import msvcrt
else:
    import select

# Shared flag and listener for GUI mode Enter detection via pynput
_enter_pressed_flag: bool = False
_pynput_listener: Optional[Any] = None

def _on_key_press(key: Any) -> None:
    global _enter_pressed_flag
    if pynput_keyboard is not None and key == pynput_keyboard.Key.enter:
        _enter_pressed_flag = True

def _ensure_pynput_listener() -> None:
    global _pynput_listener
    if pynput_keyboard is None:
        raise RuntimeError("GUI keyboard input requires a supported pynput backend")
    if _pynput_listener is None or not _pynput_listener.running:
        _pynput_listener = pynput_keyboard.Listener(on_press=_on_key_press)
        _pynput_listener.start()

def is_data() -> bool:
    if os.environ.get('GUI_MODE') == '1':
        _ensure_pynput_listener()
        return _enter_pressed_flag
    elif sys.platform == 'win32':
        return msvcrt.kbhit()
    else:
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def wait_enter_blocking() -> None:
    if os.environ.get('GUI_MODE') == '1':
        global _enter_pressed_flag
        _ensure_pynput_listener()
        _enter_pressed_flag = False
        while not _enter_pressed_flag:
            time.sleep(0.01)
        _enter_pressed_flag = False
    elif sys.platform == 'win32':
        while True:
            if msvcrt.kbhit():
                c = msvcrt.getch()
                if c == b'\r':
                    break
            time.sleep(0.01)
    else:
        while True:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                c = sys.stdin.read(1)
                if c == '\n':
                    break
            time.sleep(0.01)

def enter_pressed_nonblocking() -> bool:
    if os.environ.get('GUI_MODE') == '1':
        global _enter_pressed_flag
        _ensure_pynput_listener()
        if _enter_pressed_flag:
            _enter_pressed_flag = False
            return True
        return False
    elif sys.platform == 'win32':
        if msvcrt.kbhit():
            c = msvcrt.getch()
            if c == b'\r':
                return True
        return False
    else:
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            c = sys.stdin.read(1)
            if c == '\n':
                return True
        return False


class DataSaverConsumer(BaseConsumer):
    """
    Data saver consumer for storing device manager data.

    This consumer saves data based on device type:
    - Camera devices: Save as images (PNG/JPG)
    - Base devices: Save as NPZ files
    - Uses frame timestamp as folder name
    """

    def __init__(self, summary_shm_name: str = "device_summary_data",
                 output_dir: str = "saved_data", save_interval: float = 0.04,
                 start_idx:int=0, image_size: Optional[Tuple[int, int]] = None) -> None:
        """
        Initialize the data saver consumer.

        Args:
            summary_shm_name: Name of the summary shared memory
            output_dir: Directory to save data files
            save_interval: Interval between saves in seconds
            start_idx: Starting episode index
            image_size: Target image size (width, height) for saving camera images.
                       If None, save original size. e.g., (400, 300) for width=400, height=300
        """
        super().__init__(summary_shm_name)

        self.output_dir = output_dir
        self.save_interval = save_interval
        self.image_size = image_size

        # Data collection
        self.data_buffer: Dict[str, List[Tuple[int, Any]]] = {}
        self.frame_timestamp = None
        self.episode_counts = start_idx
        self.last_saving_timestamp = None
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def _parse_summary_header(self) -> None:
        """Parse the summary SHM header and initialize data buffers."""
        super()._parse_summary_header()

        # Initialize data buffers for each device
        for device_key in self.devices:
            self.data_buffer[device_key] = []

        logger.info(f"Initialized data saver for {len(self.devices)} devices")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Save interval: {self.save_interval}s")
        if self.image_size is not None:
            logger.info(f"Camera image will be resized to: {self.image_size} (width, height)")
        else:
            logger.info(f"Camera image will be saved at original size")

    def _save_camera_image(self, folder_path: str, device_name: str, data_array: np.ndarray, timestamp_ns: int) -> None:
        """Save camera data as JPEG image using fast imageio."""
        try:

            # Ensure data is in the correct format
            if data_array.dtype != np.uint8:
                data_array = data_array.astype(np.uint8)

            # Resize image if target size is specified
            if self.image_size is not None:
                original_shape = data_array.shape
                target_width, target_height = self.image_size
                data_array = cv2.resize(data_array, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                logger.debug(f"Resized image from {original_shape[:2]} to {data_array.shape[:2]}")

            # Generate filename
            timestamp_str = datetime.fromtimestamp(timestamp_ns / 1e9).strftime("%H%M%S_%f")[:-3]
            filename = f"{device_name}_{timestamp_str}.jpg"
            filepath = os.path.join(folder_path, filename)

            # Save image using imageio with JPEG format (fast)
            ST = time.time()
            imageio.imwrite(filepath, data_array, quality=90)
            logger.debug(f"Saved camera image: {filepath}")
            logger.info(f"Save camera image time: {time.time() - ST}s")

        except Exception as e:
            logger.error(f"Error saving camera image for {device_name}: {e}")

    def _save_base_device_data(self, folder_path: str, device_name: str, data_array: np.ndarray,
                              timestamp_ns: int, device_info: Dict[str, Any]) -> None:
        """Save base device data as NPZ file."""
        try:
            # Generate filename
            timestamp_str = datetime.fromtimestamp(timestamp_ns / 1e9).strftime("%H%M%S_%f")[:-3]
            filename = f"{device_name}_{timestamp_str}.npz"
            filepath = os.path.join(folder_path, filename)

            # Prepare metadata
            metadata = {
                'timestamp_ns': timestamp_ns,
                'device_type': device_info['type'],
                'device_class_id': device_info['id'],
                'fps': device_info['fps'],
                'data_dtype': device_info['data_dtype'],
                'hardware_latency_ms': device_info['hardware_latency_ms'],
                'shape': device_info.get('shape', data_array.shape)
            }

            # Save as NPZ
            np.savez_compressed(filepath,
                              data=data_array,
                              metadata=np.array([json.dumps(metadata)], dtype=object))

            logger.debug(f"Saved base device data: {filepath}")

        except Exception as e:
            logger.error(f"Error saving base device data for {device_name}: {e}")

    def collect_data(self) -> None:
        """Collect data from all devices."""
        current_time = time.time()
        # Read device data
        result = self.read_all_device_data(include_frame_timestamp=True)
        if result is not None:
            # Unpack frame timestamp and device data
            frame_timestamp, all_data = result
            self.frame_timestamp = frame_timestamp

            for device_key, data in all_data.items():
                timestamp_ns, data_array = data
                self.data_buffer[device_key].append((timestamp_ns, data_array))
                self.devices[device_key]['last_timestamp'] = timestamp_ns
                self.devices[device_key]['last_update_time'] = current_time

    def save_data(self) -> str:
        """Save collected data to files based on device type."""
        if not any(self.data_buffer.values()) or self.frame_timestamp is None:
            return
        # Create folder using frame timestamp
        timestamp_str = str(self.frame_timestamp)
        folder_name = f"data_{timestamp_str}"
        folder_path = os.path.join(self.output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        try:
            # Save data for each device based on its type
            for device_key, buffer_data in self.data_buffer.items():
                if not buffer_data:
                    continue

                device_info = self.devices[device_key]
                device_type = device_info['type']
                device_name = device_info['name']

                # Get the latest data
                timestamp_ns, data_array = buffer_data[-1]

                if device_type in ['OpenCVCameraDevice','HikCameraDevice']:
                    # Save camera data as image
                    self._save_camera_image(folder_path, device_name, data_array, timestamp_ns)
                elif device_type == 'BaseDevice':
                    # Save base device data as NPZ
                    self._save_base_device_data(folder_path, device_name, data_array, timestamp_ns, device_info)
                else:
                    # Save other device types as NPZ
                    self._save_base_device_data(folder_path, device_name, data_array, timestamp_ns, device_info)

            logger.info(f"Saved data to folder: {folder_path}")

            # Clear buffers after successful save
            for device_key in self.data_buffer:
                self.data_buffer[device_key].clear()

        except Exception as e:
            logger.error(f"Error saving data to {folder_path}: {e}")
        return folder_path

    def _convert_data_to_episode(self, data_folders: List[str], remove=True) -> None:
        """Organize the saved data folders into one episode .h5 file"""
        assert len(data_folders)==len(set(data_folders)), "Data folders have redundancy"
        if len(data_folders)==0: return
        try:
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
        logger.info("Start data acquisition. Ctrl+C to exit.")
        logger.info("+"*40)
        try:
            while self.running:
                logger.info("Press Enter to start data acquisition.")
                wait_enter_blocking()
                data_folders = []
                logger.info("Data acquisition started. Press Enter to stop.")
                time_episode_start = time.time()

                # Create precise timing function for save interval
                wait_for_next_save = precise_loop_timing(self.save_interval)

                while True:
                    # Collect data from devices
                    self.collect_data()

                    # Check if we should save (using frame timestamp to avoid duplicates)
                    if (self.frame_timestamp is not None and
                        self.last_saving_timestamp != self.frame_timestamp and
                        any(self.data_buffer.values())):

                        saved_data_folder = self.save_data()
                        if saved_data_folder is not None:
                            data_folders.append(saved_data_folder)
                            self.last_saving_timestamp = self.frame_timestamp

                        if len(data_folders) != len(set(data_folders)):
                            logger.error("Data folders have redundancy")

                    # Wait for next save interval using precise timing
                    wait_for_next_save()

                    if enter_pressed_nonblocking():
                        logger.info("Data acquisition stopped by user.")
                        break
                # Save any remaining data
                if all(self.data_buffer.values()) and self.last_saving_timestamp!=self.frame_timestamp:
                    saved_data_folder = self.save_data()
                    if saved_data_folder is not None and saved_data_folder not in data_folders:
                        data_folders.append(saved_data_folder)
                time_episode_end = time.time()
                logger.info(f"Time taken to collect data: {time_episode_end - time_episode_start}s at actual frequency {len(data_folders)/(time_episode_end - time_episode_start)}Hz")
                logger.info("Organizing Data into Episode...")
                # Organizing data into episode
                if len(data_folders)>0:
                    self._convert_data_to_episode(data_folders)

        except KeyboardInterrupt:
            logger.info("Data saving interrupted by user")

        finally:
            self.stop()

    def get_status(self) -> Dict[str, Any]:
        """Get status of the data saver consumer."""
        status = super().get_status()

        # Add saver-specific information
        status['output_dir'] = self.output_dir
        status['save_interval'] = self.save_interval
        status['frame_timestamp'] = self.frame_timestamp

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
    parser.add_argument("--interval", "-i", type=float, default=0.03333,
                        help="Save interval in seconds (default: 1.0)")
    parser.add_argument("--start-idx", "-st", type=int, default=0,
                        help="Start index of the episode")
    parser.add_argument("--image-size", "-is", type=str, default=None,
                        help="Target image size for camera data as 'width,height' (e.g., '400,300'). "
                             "If not specified, save at original size.")
    parser.add_argument("--status", action="store_true",
                        help="Show consumer status and exit")

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

    # Create data saver consumer
    consumer = DataSaverConsumer(
        summary_shm_name=args.summary_shm,
        output_dir=args.output_dir,
        save_interval=args.interval,
        start_idx=args.start_idx,
        image_size=image_size,
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
