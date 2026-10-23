#!/usr/bin/env python3
"""
Base Device Visualizer - Status monitoring for BaseDevice shared memory.

This visualizer focuses on device status monitoring rather than data visualization,
providing real-time status information without showing actual data values.
Supports continuous memory buffer for multiple frames.

Author: Jun Lv
"""

import time
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import argparse
from typing import List, Optional, Dict, Any
from utils.shm_utils import (
    unpack_device_header, unpack_buffer_header, unpack_frame_header,
    DEVICE_HEADER_SIZE, BUFFER_HEADER_SIZE, FRAME_HEADER_SIZE,
    connect_to_device_shm
)
from utils.logger_config import logger


class BaseVisualizer:
    """
    Base visualizer class for device status monitoring and visualization.
    Provides shared memory connection, data reading, and default status monitoring UI.
    Can be used directly for status monitoring or inherited for custom visualization.
    Supports reading from continuous memory buffer.
    """

    def __init__(self, shared_memory_name: str = "base_device_0_data", data_dtype: Any = np.float32, smoothing_window: int = 10) -> None:
        """
        Initialize the base visualizer.

        Args:
            shared_memory_name: Name of the shared memory to monitor
            data_dtype: Expected data type of the numpy array
            smoothing_window: Number of samples for moving average smoothing
        """
        self.shared_memory_name = shared_memory_name
        self.data_dtype = data_dtype
        self.shared_memory = None

        # Status tracking
        self.last_timestamp = None
        self.fps_estimate = 0.0
        self.update_count = 0
        self.start_time = time.time()
        self.last_update_time = None

        # Smoothing for better visualization
        self.fps_history = []  # Store recent FPS values for smoothing
        self.latency_history = []  # Store recent latency values for smoothing
        self.smoothing_window = smoothing_window  # Number of samples for moving average

        # Connection status
        self.connected = False
        self.last_successful_read = None

        # Matplotlib setup - default status monitoring UI
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.fig.suptitle('Device Status Monitor', fontsize=16, fontweight='bold')

        # Remove axes for clean status display
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')

    def connect(self) -> bool:
        """Connect to the shared memory using unified interface (read-only for visualizer)."""
        try:
            # Extract device info from SHM name (e.g., "base_device_0_data" -> "base_device", 0)
            if self.shared_memory_name.endswith('_data'):
                base_name = self.shared_memory_name[:-5]  # Remove '_data'
                parts = base_name.rsplit('_', 1)
                if len(parts) == 2:
                    device_name, device_class_id_str = parts
                    device_class_id = int(device_class_id_str)

                    # Use unified SHM interface for read-only connection
                    self.shared_memory = connect_to_device_shm(device_name, device_class_id, read_only=True)
                    self.connected = True
                    logger.info(f"Connected to shared memory: {self.shared_memory_name} (READ-ONLY)")
                    return True
                else:
                    logger.error(f"Invalid device SHM name format: {self.shared_memory_name}")
                    self.connected = False
                    return False
            else:
                logger.error(f"Device SHM name must end with '_data': {self.shared_memory_name}")
                self.connected = False
                return False

        except FileNotFoundError:
            self.connected = False
            logger.warning(f"Shared memory {self.shared_memory_name} not found.")
            logger.warning("Make sure the base device server is running.")
            return False
        except Exception as e:
            self.connected = False
            logger.error(f"Error connecting to shared memory: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from shared memory and clean up resources."""
        if self.shared_memory:
            try:
                self.shared_memory.close()
                logger.info(f"Disconnected from shared memory: {self.shared_memory_name}")
            except Exception as e:
                logger.error(f"Error disconnecting from shared memory: {e}")
            finally:
                self.shared_memory = None
                self.connected = False

    def read_latest_frame(self) -> tuple:
        """
        Read the latest frame from shared memory buffer.

        Returns:
            tuple: (timestamp_ns, data_array) or (None, None) if error
        """
        return self.read_frame_at_index(-1)

    def _parse_buffer_header(self) -> tuple:
        """
        Parse the buffer header from shared memory.

        Returns:
            tuple: (buffer_size, current_frames_count, write_index, shape, frame_size)
                   or (None, None, None, None, None) if error
        """
        if not self.shared_memory:
            return None, None, None, None, None

        try:
            buf = self.shared_memory.buf

            # Parse device header
            device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
            shape = device_header['shape']
            frame_size = device_header['frame_size']

            # Parse buffer header
            buffer_header = unpack_buffer_header(buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE+BUFFER_HEADER_SIZE])
            buffer_size = buffer_header['buffer_size']
            current_frames_count = buffer_header['current_frames_count']
            write_index = buffer_header['write_index']

            return buffer_size, current_frames_count, write_index, shape, frame_size

        except Exception as e:
            logger.error(f"Error parsing buffer header: {e}")
            return None, None, None, None, None

    def _read_frame_data(self, frame_offset: int, shape: tuple, frame_size: int) -> tuple:
        """
        Read frame data from the specified offset.

        Args:
            frame_offset: Offset to the frame in shared memory
            shape: Shape of the data array
            frame_size: Size of the frame in bytes

        Returns:
            tuple: (timestamp_ns, data_array) or (None, None) if error
        """
        try:
            # Read frame header
            frame_header = unpack_frame_header(self.shared_memory.buf[frame_offset:frame_offset+FRAME_HEADER_SIZE])
            timestamp_ns = frame_header['timestamp_ns']

            # Read frame data
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_end = data_start + frame_size - FRAME_HEADER_SIZE

            # Extract data: DataChannel.buf returns raw buffer, slicing works directly
            sliced_buf = self.shared_memory.buf[data_start:data_end]
            data_array = np.frombuffer(sliced_buf, dtype=self.data_dtype)

            # Reshape if needed
            if len(shape) > 0:
                data_array = data_array.reshape(shape)

            return timestamp_ns, data_array

        except Exception as e:
            logger.error(f"Error reading frame data: {e}")
            return None, None

    def read_frame_at_index(self, index: int) -> tuple:
        """
        Read frame at specific index from buffer.

        Args:
            index: Frame index (negative for latest frames, e.g., -1 for latest)

        Returns:
            tuple: (timestamp_ns, data_array) or (None, None) if error
        """
        # Parse buffer header
        header_info = self._parse_buffer_header()
        if header_info[0] is None:  # Check if parsing failed
            return None, None

        buffer_size, current_frames_count, write_index, shape, frame_size = header_info

        # Check if buffer has data
        if current_frames_count == 0:
            return None, None

        # Calculate actual index
        if index < 0:
            actual_index = (write_index + index) % buffer_size
        else:
            actual_index = index % buffer_size

        # Check if the requested frame has been written
        if index < 0:
            # For negative indices (latest frames), check if we have enough frames
            frames_available = min(current_frames_count, buffer_size)
            if abs(index) > frames_available:
                logger.warning(f"Requested frame {index} but only {frames_available} frames available")
                return None, None
        else:
            # For positive indices, check if the frame has been written
            if actual_index >= current_frames_count:
                logger.warning(f"Requested frame {index} (actual: {actual_index}) but only {current_frames_count} frames written")
                return None, None

        frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + actual_index * frame_size

        # Read frame data
        return self._read_frame_data(frame_offset, shape, frame_size)

    def get_buffer_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current buffer state.

        Returns:
            dict: Buffer information including current_frames_count, buffer_size, write_index, etc.
        """
        header_info = self._parse_buffer_header()
        if header_info[0] is None:
            return None

        buffer_size, current_frames_count, write_index, shape, frame_size = header_info

        return {
            'buffer_size': buffer_size,
            'current_frames_count': current_frames_count,
            'write_index': write_index,
            'shape': shape,
            'frame_size': frame_size,
            'frames_available': min(current_frames_count, buffer_size),
            'buffer_full': current_frames_count >= buffer_size
        }

    def calculate_fps(self, timestamp_ns: int) -> None:
        """Calculate FPS based on timestamp with smoothing."""
        if self.last_timestamp is not None:
            time_diff = (timestamp_ns - self.last_timestamp) / 1e9
            if time_diff > 0:
                current_fps = 1.0 / time_diff
                # Add to history for smoothing
                self.fps_history.append(current_fps)
                if len(self.fps_history) > self.smoothing_window:
                    self.fps_history.pop(0)  # Remove oldest value

                # Calculate smoothed FPS
                if self.fps_history:
                    self.fps_estimate = sum(self.fps_history) / len(self.fps_history)
        self.last_timestamp = timestamp_ns

    def calculate_latency(self, timestamp_ns: int) -> float:
        """Calculate latency in milliseconds with smoothing."""
        current_time_ns = time.time_ns()
        latency_ns = current_time_ns - timestamp_ns
        current_latency_ms = latency_ns / 1e6  # Convert to milliseconds

        # Add to history for smoothing
        self.latency_history.append(current_latency_ms)
        if len(self.latency_history) > self.smoothing_window:
            self.latency_history.pop(0)  # Remove oldest value

        # Return smoothed latency
        if self.latency_history:
            return sum(self.latency_history) / len(self.latency_history)
        else:
            return current_latency_ms

    def update_status(self, frame: Any) -> List:
        """Update the status display."""
        timestamp_ns, data_array = self.read_latest_frame()
        current_time = time.time()

        # Clear previous status
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')

        if timestamp_ns is not None and data_array is not None:
            # Update counters
            self.calculate_fps(timestamp_ns)
            self.update_count += 1
            self.last_successful_read = current_time
            self.last_update_time = current_time

            # Convert timestamp to readable format
            timestamp_s = timestamp_ns / 1e9
            dt = datetime.fromtimestamp(timestamp_s)
            time_str = dt.strftime("%H:%M:%S.%f")[:-3]

            # Calculate latency
            latency_ms = self.calculate_latency(timestamp_ns)

            # Calculate uptime
            uptime = current_time - self.start_time

            # Create data info from array
            data_info = {
                'shape': data_array.shape,
                'size': data_array.size,
                'data_type': str(self.data_dtype),
                'memory_bytes': data_array.nbytes,
                'min_value': float(np.min(data_array)),
                'max_value': float(np.max(data_array)),
                'mean_value': float(np.mean(data_array)),
                'std_value': float(np.std(data_array))
            }

            # Connection status
            status_color = 'green'
            status_text = '[CONNECTED]'
        else:
            # No data available
            status_color = 'red'
            status_text = '[NO DATA]'
            time_str = 'N/A'
            latency_ms = 0
            uptime = current_time - self.start_time
            data_info = {
                'shape': 'N/A',
                'size': 0,
                'data_type': str(self.data_dtype),
                'memory_bytes': 0,
                'min_value': 0,
                'max_value': 0,
                'mean_value': 0,
                'std_value': 0
            }

        # Check connection health
        if self.last_successful_read and (current_time - self.last_successful_read) > 5.0:
            status_color = 'orange'
            status_text = '[CONNECTION ISSUES]'

        # Get buffer information
        buffer_info = self.get_buffer_info()

        # Format uptime
        uptime_str = f"{int(uptime//3600):02d}:{int((uptime%3600)//60):02d}:{int(uptime%60):02d}"

        # Create status display
        status_info = f"""
        DEVICE STATUS MONITOR
        ═══════════════════════════════════════

        CONNECTION
        Status: {status_text}
        Shared Memory: {self.shared_memory_name}
        Uptime: {uptime_str}
        Updates Received: {self.update_count:,}

        BUFFER INFORMATION
        Buffer Size: {buffer_info['buffer_size'] if buffer_info else 'N/A'} frames
        Frames Available: {buffer_info['frames_available'] if buffer_info else 'N/A'}
        Current Count: {buffer_info['current_frames_count'] if buffer_info else 'N/A'}
        Write Index: {buffer_info['write_index'] if buffer_info else 'N/A'}
        Buffer Full: {buffer_info['buffer_full'] if buffer_info else 'N/A'}

        PERFORMANCE
        FPS: {self.fps_estimate:.1f}
        Latency: {latency_ms:.1f} ms
        Last Update: {time_str}

        DATA INFORMATION
        Shape: {data_info['shape']}
        Elements: {data_info['size']:,}
        Data Type: {data_info['data_type']}
        Memory Size: {data_info['memory_bytes']:,} bytes

        STATISTICS (Current Frame)
        Range: [{data_info['min_value']:.3f}, {data_info['max_value']:.3f}]
        Mean: {data_info['mean_value']:.3f}
        Std Dev: {data_info['std_value']:.3f}
        """

        # Display status text
        self.ax.text(0.05, 0.95, status_info.strip(),
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='left',
                    fontsize=12,
                    fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5",
                             facecolor='white',
                             alpha=0.9,
                             edgecolor=status_color,
                             linewidth=2),
                    color='black')

        return []

    def start_visualization(self, update_interval: int = 1000) -> None:
        """
        Start status monitoring visualization.

        Args:
            update_interval: Update interval in milliseconds
        """
        # Try to connect initially
        self.connect()

        logger.info("Starting Base Device Status Monitor...")
        logger.info(f"Monitoring: {self.shared_memory_name}")
        logger.info("Close the window to stop monitoring.")
        logger.info("-" * 50)

        # Set up the animation
        ani = animation.FuncAnimation(self.fig, self.update_status, interval=update_interval,
                                    blit=False, cache_frame_data=False)

        # Show the plot
        plt.tight_layout()
        plt.show()

        # Clean up
        self.disconnect()

    def __del__(self) -> None:
        """Destructor to ensure proper cleanup."""
        try:
            self.disconnect()
        except:
            pass


def main() -> None:
    """Main function to run the base device status monitor."""
    parser = argparse.ArgumentParser(description="Base Device Status Monitor")
    parser.add_argument("--shared-memory", "-s", default="base_device_0_data",
                        help="Shared memory name (default: base_device_0_data)")
    parser.add_argument("--dtype", "-d", default="float32",
                        choices=['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
                                'float32', 'float64'],
                        help="Data type (default: float32)")
    parser.add_argument("--interval", "-i", type=int, default=1,
                        help="Update interval in ms (default: 1)")
    parser.add_argument("--smoothing", "-sm", type=int, default=10,
                        help="Smoothing window size for FPS and latency (default: 10)")

    args = parser.parse_args()

    # Import common utilities
    from utils.shm_utils import get_dtype

    logger.info(f"Starting Base Device Status Monitor...")
    logger.info(f"Target: {args.shared_memory}")
    logger.info(f"Data type: {args.dtype}")
    logger.info(f"Update interval: {args.interval}ms")

    monitor = BaseVisualizer(
        shared_memory_name=args.shared_memory,
        data_dtype=get_dtype(args.dtype),
        smoothing_window = args.smoothing
    )

    monitor.start_visualization(update_interval=args.interval)


if __name__ == "__main__":
    main()
