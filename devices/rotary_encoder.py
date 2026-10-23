#!/usr/bin/env python3
"""
Rotary Encoder Device for reading gripper width data via serial communication.

This device reads angle data from a rotary encoder through serial communication,
converts it to gripper width using a calibration file, and writes the width data
to shared memory for other processes to consume.

Author: Han Xue, Jun Lv
"""

import time
import serial
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Mapping, Union
from scipy.interpolate import interp1d
from pathlib import Path
from utils.logger_config import logger

from devices.base import BaseDevice
from devices.config import StateFeedbackConfig
from utils.shm_utils import get_dtype
from utils.time_control import precise_loop_timing


class RotaryEncoderDevice(BaseDevice):
    """
    Rotary encoder device that reads angle data via serial communication
    and converts it to gripper width using calibration data.

    This device connects to a rotary encoder through serial communication,
    reads angle data, converts it to gripper width using interpolation from
    a calibration file, and writes the width data to shared memory.
    """

    def __init__(self,
                 device_class_id: int = 0,
                 *,
                 state_feedback: StateFeedbackConfig | Mapping[str, Any],
                 port: str = "/dev/ttyUSB0",
                 baudrate: int = 57600,
                 calibration_file: str = "data/gripper_width_calibration.csv",
                 ) -> None:
        """
        Initialize the rotary encoder device.

        Args:
            device_class_id: Unique identifier for this device instance
            port: Serial port name (default: "/dev/ttyUSB0")
            baudrate: Serial communication baudrate (default: 57600)
            calibration_file: Path to CSV calibration file (default: "data/gripper_width_calibration.csv")
            fps: Frames per second for data reading (default: 10.0)
            buffer_size: Number of frames to store in buffer (default: 10)
            hardware_latency_ms: Hardware latency in milliseconds (default: 5.0)
            data_dtype: Data type for width data (default: np.float64)
        """
        # Gripper width data: single value (width in mm)
        self.device_name = "RotaryEncoderDevice"

        super().__init__(
            device_class_id=device_class_id,
            state_feedback=state_feedback,
            device_name=self.device_name,
        )

        # Serial communication parameters
        self.port: str = port
        self.baudrate: int = baudrate
        self.serial_conn: Optional[serial.Serial] = None

        # Communication data (from reference script)
        self.send_data: bytes = bytes.fromhex("01 03 00 41 00 01 d4 1e")

        # Statistics
        self.send_count: int = 0
        self.receive_count: int = 0
        self.error_count: int = 0
        self.last_angle: float = 0.0
        self.last_width: float = 0.0

        # Calibration
        self.calibration_file: str = calibration_file
        self.angle_to_width_func: Optional[interp1d] = None

        # Initialize calibration and serial connection
        self._load_calibration()
        self._initialize_serial()

    def _load_calibration(self) -> None:
        """Load calibration data from CSV file and create interpolation function."""
        try:
            # Check if calibration file exists
            cal_path = Path(self.calibration_file)
            if not cal_path.exists():
                logger.error(f"Calibration file not found: {self.calibration_file}")
                raise FileNotFoundError(f"Calibration file not found: {self.calibration_file}")

            # Load CSV data
            df = pd.read_csv(self.calibration_file)
            logger.info(f"Loaded calibration file: {self.calibration_file}")
            logger.info(f"Calibration data shape: {df.shape}")

            # Extract angle and width columns
            # Handle possible Chinese characters in column names
            angle_col = None
            width_col = None

            for col in df.columns:
                if 'angle' in col.lower() or 'deg' in col.lower():
                    angle_col = col
                elif 'width' in col.lower() or 'mm' in col.lower():
                    width_col = col

            if angle_col is None or width_col is None:
                logger.error(f"Could not find angle and width columns in {self.calibration_file}")
                logger.info(f"Available columns: {list(df.columns)}")
                raise ValueError("Could not find angle and width columns")

            # Remove any rows with NaN values
            df_clean = df[[angle_col, width_col]].dropna()

            angles = df_clean[angle_col].values
            widths = df_clean[width_col].values

            logger.info(f"Calibration range: Angle [{angles.min():.3f}, {angles.max():.3f}] deg")
            logger.info(f"                   Width [{widths.min():.1f}, {widths.max():.1f}] mm")

            # Create interpolation function
            # Use 'linear' for simple interpolation, can change to 'cubic' for smoother
            self.angle_to_width_func = interp1d(
                angles, widths,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )

            logger.info("Calibration interpolation function created successfully")

        except Exception as e:
            logger.error(f"Error loading calibration file: {e}")
            # Create a dummy linear function as fallback
            self.angle_to_width_func = lambda x: np.clip(x * -3.0 + 50.0, 0.0, 100.0)
            logger.warning("Using fallback linear calibration function")

    def _initialize_serial(self) -> bool:
        """Initialize serial connection."""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.02
            )
            logger.info(f"Serial connection established: {self.port} @ {self.baudrate} baud")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize serial connection: {e}")
            logger.warning("Device will run with zero width data")
            self.serial_conn = None
            return False

    def _send_command(self) -> bool:
        """Send command to serial port."""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(self.send_data)
                self.send_count += 1
                return True
            except Exception as e:
                logger.error(f"Error sending command: {e}")
                self.error_count += 1
                return False
        return False

    def _read_response(self) -> Optional[bytes]:
        """Read response from serial port."""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                # Try to read data
                data = self.serial_conn.read(7)  # Read up to 7 bytes
                if data:
                    self.receive_count += 1
                    return data
            except Exception as e:
                logger.error(f"Error reading response: {e}")
                self.error_count += 1
                return None
        return None

    def _calculate_angle(self, data_bytes: bytes) -> Optional[float]:
        """
        Calculate angle value from received data (based on reference script).

        Args:
            data_bytes: Received data from serial port

        Returns:
            float: Calculated angle value in degrees, or None if invalid
        """
        try:
            if len(data_bytes) >= 7:
                # Extract angle data (4th and 5th bytes)
                angle_high = data_bytes[3]
                angle_low = data_bytes[4]
                angle_raw = (angle_high << 8) | angle_low
                mask = 0b111111111111
                data_bit = 0b100000000000
                angle_raw = angle_raw & mask

                if (angle_raw & data_bit):
                    angle_raw = -((~angle_raw & mask) + 1)

                # Calculate angle: 360 * raw_value / 4096
                angle = 360.0 * angle_raw / 4096.0
                return angle
        except Exception as e:
            logger.error(f"Error calculating angle: {e}")
            self.error_count += 1
            return None

        return None

    def _angle_to_width(self, angle: float) -> float:
        """
        Convert angle to gripper width using calibration interpolation.

        Args:
            angle: Angle value in degrees

        Returns:
            float: Gripper width in mm
        """
        if self.angle_to_width_func is None:
            return 0.0

        try:
            width = float(self.angle_to_width_func(angle))
            # Ensure width is non-negative
            return max(0.0, width)
        except Exception as e:
            logger.error(f"Error converting angle to width: {e}")
            return 0.0

    def _get_width_data(self) -> Optional[float]:
        """
        Get current gripper width data from the encoder.

        Returns:
            float: Current gripper width in mm, or None if invalid
        """
        if not self.serial_conn:
            return None

        # Send command and read response
        if self._send_command():
            response = self._read_response()
            if response:
                angle = self._calculate_angle(response)
                if angle is not None:
                    self.last_angle = angle
                    width = self._angle_to_width(angle)
                    self.last_width = width
                    return width

        return None

    def start_server(self) -> None:
        """Start the rotary encoder device server."""
        if self.running:
            logger.warning(f"Device {self.device_name}_{self.device_class_id} is already running")
            return

        logger.info(f"Starting {self.device_name}_{self.device_class_id} server...")
        logger.info(f"Serial port: {self.port} @ {self.baudrate} baud")
        logger.info(f"Calibration file: {self.calibration_file}")
        self.running = True

        # Create shared memory
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")

        logger.info(f"Server started. Shared memory: {self.shared_memory_name}")
        logger.info(f"Buffer configuration: {self.buffer_size} frames, {self.frame_size:,} bytes per frame")

        # Create precise timing function
        wait_for_next_iteration = precise_loop_timing(self.update_interval)

        # Main data reading loop
        while self.running:
            try:
                # Get width data and timestamp
                width_data = self._get_width_data()
                timestamp_ns = time.time_ns()

                if width_data is not None:
                    numpy_data_dtype = get_dtype(self.data_dtype)
                    width_array = np.array([width_data], dtype=numpy_data_dtype)
                    self._write_array_to_shm_with_timestamp(width_array, timestamp_ns)
                else:
                    # Write zero data if no valid reading
                    numpy_data_dtype = get_dtype(self.data_dtype)
                    zero_data = np.zeros(self.data_shape, dtype=numpy_data_dtype)
                    self._write_array_to_shm_with_timestamp(zero_data, timestamp_ns)

                # Wait for next iteration using precise timing
                wait_for_next_iteration()

            except Exception as e:
                logger.error(f"Error in width data reading: {e}")
                break

    def stop_server(self) -> None:
        """Stop the rotary encoder device server."""
        if not self.running:
            return

        logger.info(f"Stopping {self.device_name}_{self.device_class_id} server...")
        self.running = False

        # Clean up serial connection
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
                logger.info("Serial connection closed")
            except Exception as e:
                logger.error(f"Error closing serial connection: {e}")

        self._cleanup_shared_memory()
        logger.info("Server stopped")

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        info = super().get_device_info()
        info.update({
            "port": self.port,
            "baudrate": self.baudrate,
            "calibration_file": self.calibration_file,
            "serial_connected": self.serial_conn is not None and self.serial_conn.is_open if self.serial_conn else False,
            "send_count": self.send_count,
            "receive_count": self.receive_count,
            "error_count": self.error_count,
            "last_angle": self.last_angle,
            "last_width": self.last_width
        })
        return info

    def __del__(self) -> None:
        """Destructor to ensure proper cleanup."""
        try:
            if hasattr(self, 'serial_conn') and self.serial_conn:
                self.serial_conn.close()
        except:
            pass
        super().__del__()


def main() -> None:
    """Main function to run the rotary encoder device server."""
    import argparse

    parser = argparse.ArgumentParser(description="Rotary encoder device server")
    parser.add_argument("--device-class-id", "-d", type=int, default=0,
                        help="Device ID (default: 0)")
    parser.add_argument("--port", "-p", type=str, default="/dev/ttyUSB0",
                        help="Serial port (default: /dev/ttyUSB0)")
    parser.add_argument("--baudrate", "-b", type=int, default=57600,
                        help="Baudrate (default: 57600)")
    parser.add_argument("--calibration-file", "-c", type=str,
                        default="../data/gripper_width_calibration.csv",
                        help="Calibration file path (default: data/gripper_width_calibration.csv)")
    parser.add_argument("--fps", "-f", type=float, default=40.0,
                        help="Frames per second (default: 40.0)")
    parser.add_argument("--buffer-size", "-s", type=int, default=10,
                        help="Buffer size in frames (default: 10)")
    parser.add_argument("--hardware-latency", "-l", type=float, default=5.0,
                        help="Hardware latency in milliseconds (default: 5.0)")
    parser.add_argument("--data-dtype", type=str, default="float64",
                        help="Data type for width data (default: float64)")

    args = parser.parse_args()

    # Create device with parsed arguments
    device = RotaryEncoderDevice(
        device_class_id=args.device_class_id,
        state_feedback={
            "fps": args.fps,
            "buffer_size": args.buffer_size,
            "hardware_latency_ms": args.hardware_latency,
            "format": {
                "shape": [1],
                "dtype": args.data_dtype,
                "type": ["gripper"],
            },
        },
        port=args.port,
        baudrate=args.baudrate,
        calibration_file=args.calibration_file,
    )

    logger.info("Rotary Encoder Device Server")
    logger.info("============================")
    logger.info(f"Device ID: {args.device_class_id}")
    logger.info(f"Serial Port: {args.port}")
    logger.info(f"Baudrate: {args.baudrate}")
    logger.info(f"Calibration File: {args.calibration_file}")
    logger.info(f"FPS: {args.fps}")
    logger.info(f"Buffer size: {args.buffer_size} frames")
    logger.info(f"Hardware latency: {args.hardware_latency} ms")
    logger.info(f"Data type: {args.data_dtype}")

    try:
        logger.info("Device server is running. Press Ctrl+C to stop...")
        logger.info(f"Device info: {device.get_device_info()}")
        device.start_server()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        device.stop_server()


if __name__ == "__main__":
    main()
