#!/usr/bin/env python3
"""
Vive Tracker Latency Calibration Script

This script measures latency between Rizon robot and Vive Tracker
using cross-correlation analysis. The robot moves in a sinusoidal
pattern while the Vive Tracker continuously reads pose data.

Author: Zixi Ying
"""

import sys
import os
import time
import numpy as np
import yaml
import argparse
import threading
import queue
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from devices.vive_tracker import ViveTrackerDevice
from devices.robot import RizonRobot
from utils.logger_config import logger
from utils.latency_util import get_latency
from utils.time_control import precise_wait, precise_sleep


class ViveTrackerLatencyCalibrator:
    """
    Calibrator for measuring latency between Rizon robot and Vive Tracker.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialize the calibrator.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()

        # Initialize devices
        self.vive_tracker: Optional[ViveTrackerDevice] = None
        self.robot: Optional[RizonRobot] = None

        # Test data storage - now using actual robot state
        self.t_robot: List[float] = []  # Timestamps for robot state
        self.x_robot: List[float] = []  # Actual robot position on target axis
        self.t_vive: List[float] = []   # Timestamps for Vive data
        self.x_vive: List[float] = []   # Vive position on target axis

        # Test parameters
        self.dt = 1.0 / self.config['test']['frequency']
        self.command_latency = self.config['test']['command_latency']
        self.robot_axis = self.config['test']['robot_axis']
        self.vive_axis = self.config['test']['vive_axis']
        self.amplitude = self.config['test']['amplitude']
        self.duration = self.config['test']['duration']

        # Motion parameters
        self.max_pos_speed = self.config['motion']['max_pos_speed']
        self.max_rot_speed = self.config['motion']['max_rot_speed']
        self.cube_diag = self.config['motion']['cube_diag']

        # Threading and queue support
        self.command_queue = queue.Queue()
        self.vive_queue = queue.Queue()
        self.running = False
        self.command_thread = None
        self.vive_thread = None
        self.robot_state_thread = None

        # Threading configuration
        vive_config = self.config["vive_tracker"]
        state_feedback = vive_config.get("state_feedback") or {}
        self.vive_read_frequency = vive_config.get(
            "read_frequency",
            state_feedback.get("fps", vive_config.get("fps", 90.0)),
        )
        self.robot_state_frequency = self.config.get('robot', {}).get('state_frequency', 100)  # Hz
        self.command_queue_timeout = self.config.get('threading', {}).get('command_queue_timeout', 0.1)
        self.thread_join_timeout = self.config.get('threading', {}).get('thread_join_timeout', 2.0)

        logger.info("Vive Tracker Latency Calibrator initialized")
        logger.info(f"Test frequency: {self.config['test']['frequency']} Hz")
        logger.info(f"Test duration: {self.duration} seconds")
        logger.info(f"Robot motion axis: {self.robot_axis}")
        logger.info(f"Vive reading axis: {self.vive_axis}")
        logger.info(f"Robot state sampling frequency: {self.robot_state_frequency} Hz")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _command_thread_worker(self) -> None:
        """Worker thread for sending robot commands at precise timing."""
        logger.info("Command thread started")
        while self.running:
            try:
                # Get command from queue with timeout
                command_data = self.command_queue.get(timeout=self.command_queue_timeout)
                target_pose, t_command_target = command_data

                # Wait until command target time
                current_time = time.time()
                if t_command_target > current_time:
                    precise_wait(t_command_target, time_func=time.time)

                # Send command to robot
                self.robot.robot.SendCartesianMotionForce(target_pose)
                logger.debug(f"Command sent at {time.time():.6f}, target time was {t_command_target:.6f}")

            except queue.Empty:
                # No commands in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error in command thread: {e}")
                break

        logger.info("Command thread stopped")

    def _vive_thread_worker(self) -> None:
        """Worker thread for high-frequency Vive Tracker reading."""
        logger.info("Vive Tracker thread started")
        vive_read_freq = self.vive_read_frequency
        vive_dt = 1.0 / vive_read_freq

        logger.info(f"Vive Tracker reading frequency: {vive_read_freq} Hz (dt: {vive_dt*1000:.2f} ms)")

        while self.running:
            try:
                start_time = time.time()

                # Read Vive Tracker data
                vive_pose = self.vive_tracker._get_pose_data()
                if vive_pose is not None:
                    current_time = time.time()
                    self.t_vive.append(current_time)
                    self.x_vive.append(vive_pose[self.vive_axis])
                    logger.debug(f"Vive data read: {vive_pose[self.vive_axis]:.6f} at {current_time:.6f}")

                # Calculate sleep time to maintain desired frequency
                elapsed = time.time() - start_time
                sleep_time = max(0, vive_dt - elapsed)
                if sleep_time > 0:
                    precise_sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in Vive Tracker thread: {e}")
                break

        logger.info("Vive Tracker thread stopped")

    def _robot_state_thread_worker(self) -> None:
        """Worker thread for high-frequency robot state reading."""
        logger.info("Robot state thread started")
        state_read_freq = self.robot_state_frequency
        state_dt = 1.0 / state_read_freq

        logger.info(f"Robot state reading frequency: {state_read_freq} Hz (dt: {state_dt*1000:.2f} ms)")

        while self.running:
            try:
                start_time = time.time()

                # Get current robot state
                current_time = time.time()

                # Extract position from TCP pose (format: [x, y, z, rx, ry, rz])
                tcp_pose = self.robot.robot.states().tcp_pose

                # Store data for the target axis
                self.t_robot.append(current_time)
                self.x_robot.append(tcp_pose[self.robot_axis]-self.robot.home_pose[self.robot_axis])

                logger.debug(f"Robot state read: {tcp_pose[self.robot_axis]:.6f} at {current_time:.6f}")

                # Calculate sleep time to maintain desired frequency
                elapsed = time.time() - start_time
                sleep_time = max(0, state_dt - elapsed)
                if sleep_time > 0:
                    precise_sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in robot state thread: {e}")
                break

        logger.info("Robot state thread stopped")

    def initialize_devices(self) -> bool:
        """Initialize Vive Tracker and Rizon Robot devices."""
        try:
            logger.info("Initializing devices...")

            # Initialize Vive Tracker
            logger.info("Initializing Vive Tracker...")
            vive_config = self.config["vive_tracker"]
            tracker_roles = vive_config.get("tracker_roles")
            state_feedback = vive_config.get("state_feedback")
            if state_feedback is None:
                state_feedback = {
                    "fps": float(vive_config.get("fps", 90.0)),
                    "buffer_size": int(vive_config.get("buffer_size", 100)),
                    "hardware_latency_ms": float(
                        vive_config.get("hardware_latency_ms", 0.0)
                    ),
                    "format": {
                        "shape": [len(tracker_roles or [None]), 7],
                        "dtype": str(vive_config.get("data_dtype", "float64")),
                        "type": ["xyz", "qxqyqzqw"] * len(tracker_roles or [None]),
                    },
                }
            self.vive_tracker = ViveTrackerDevice(
                device_class_id=0,
                state_feedback=state_feedback,
                tracker_roles=tracker_roles,
            )
            logger.info("Vive Tracker initialized successfully")

            # Initialize Rizon Robot
            logger.info("Initializing Rizon Robot...")
            robot_config = self.config['robot']
            self.robot = RizonRobot(
                device_class_id=1,
                robot_sn=robot_config['robot_sn'],
                state_feedback=robot_config['state_feedback'],
                control=robot_config['control'],
                gripper=robot_config.get('gripper'),
                motion_limits=robot_config.get('motion_limits'),
                home=robot_config['home'],
                signal_files={
                    "robot_home_command_pattern": "/tmp/{device_name}_{device_class_id}_home_command",
                    "policy_runner_reset_command": "/tmp/policy_runner_reset_command",
                    "reset_signal": "/tmp/policy_connector_reset_signal",
                },
            )
            assert not self.robot.is_dual_arm, "Dual arm robot forbidden"

            # Connect to robot
            logger.info("Connecting to robot...")
            if not self.robot._connect_robot():
                raise RuntimeError("Failed to connect to robot")

            # Switch to EEF control
            logger.info("Switching to EEF control mode...")
            if not self.robot._switch_to_eef_control():
                raise RuntimeError("Failed to switch to EEF control mode")

            # Set home pose
            logger.info("Setting home pose...")
            if not self.robot._set_home_pose():
                raise RuntimeError("Failed to set home pose")

            logger.info("All devices initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize devices: {e}")
            return False

    def run_calibration(self) -> bool:
        """Run the latency calibration test using real robot state."""
        try:
            logger.info("Starting latency calibration test...")
            logger.info("Press Ctrl+C to stop early")

            # Get initial robot state
            robot_state = self.robot.get_robot_state()
            if self.robot.is_dual_arm:
                # For dual arm, use left arm pose
                home_pose = self.robot.home_pose_left
            else:
                home_pose = self.robot.home_pose

            if home_pose is None:
                raise RuntimeError("Home pose not set")

            # Convert to Flexiv format [x,y,z,qw,qx,qy,qz]
            target_pose = [
                home_pose[0], home_pose[1], home_pose[2],
                home_pose[6], home_pose[3], home_pose[4], home_pose[5]
            ]

            # Start worker threads
            self.running = True
            self.command_thread = threading.Thread(target=self._command_thread_worker, daemon=True)
            self.vive_thread = threading.Thread(target=self._vive_thread_worker, daemon=True)
            self.robot_state_thread = threading.Thread(target=self._robot_state_thread_worker, daemon=True)

            self.command_thread.start()
            self.vive_thread.start()
            self.robot_state_thread.start()

            logger.info("Worker threads started")

            t_start = time.time()
            iter_idx = 0
            max_iterations = int(self.duration * self.config['test']['frequency'])

            while iter_idx < max_iterations:
                try:
                    # Calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * self.dt
                    t_command_target = t_cycle_end + self.dt

                    # Generate sinusoidal motion command
                    t_current = time.time() - t_start
                    sin_signal = self.amplitude * np.sin(2 * np.pi * 0.3 * t_current)

                    # Apply motion to target axis
                    if self.robot_axis < 3:  # Position axis
                        target_pose[self.robot_axis] = home_pose[self.robot_axis] + sin_signal
                    else:  # Rotation axis (convert to quaternion)
                        target_pose[self.robot_axis] = home_pose[self.robot_axis] + sin_signal

                    # Add command to queue for worker thread
                    self.command_queue.put((target_pose.copy(), t_command_target))

                    # Wait for cycle end
                    precise_wait(t_cycle_end, time_func=time.time)
                    iter_idx += 1

                    if iter_idx % 100 == 0:
                        logger.info(f"Test progress: {iter_idx}/{max_iterations}")
                        logger.info(f"Robot samples: {len(self.t_robot)}, Vive samples: {len(self.t_vive)}")

                except KeyboardInterrupt:
                    logger.info("Test interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error during test iteration {iter_idx}: {e}")
                    break

            # Stop worker threads
            self.running = False

            # Wait for threads to finish
            if self.command_thread and self.command_thread.is_alive():
                self.command_thread.join(timeout=self.thread_join_timeout)
            if self.vive_thread and self.vive_thread.is_alive():
                self.vive_thread.join(timeout=self.thread_join_timeout)
            if self.robot_state_thread and self.robot_state_thread.is_alive():
                self.robot_state_thread.join(timeout=self.thread_join_timeout)

            logger.info(f"Calibration test completed. Collected {len(self.t_robot)} robot samples.")
            logger.info(f"Vive Tracker samples collected: {len(self.t_vive)}")
            return True

        except Exception as e:
            logger.error(f"Failed to run calibration: {e}")
            return False

    def analyze_latency(self) -> Tuple[float, Dict[str, Any]]:
        """Analyze the collected data to calculate latency using real robot state."""
        try:
            logger.info("Analyzing latency using real robot state...")

            if len(self.t_robot) == 0 or len(self.t_vive) == 0:
                raise RuntimeError("No data collected for analysis")

            # Convert to numpy arrays
            self.t_robot = np.array(self.t_robot)
            self.x_robot = np.array(self.x_robot)
            self.t_vive = np.array(self.t_vive)
            self.x_vive = np.array(self.x_vive)

            self.x_robot = -self.x_robot
            # Normalize signals for better correlation
            self.x_robot = (self.x_robot - np.mean(self.x_robot)) / np.std(self.x_robot)
            self.x_vive = (self.x_vive - np.mean(self.x_vive)) / np.std(self.x_vive)

            # Debug: Print data statistics
            logger.info(f"Robot data: {len(self.t_robot)} samples, time range: {self.t_robot[0]:.3f} to {self.t_robot[-1]:.3f}")
            logger.info(f"Vive data: {len(self.t_vive)} samples, time range: {self.t_vive[0]:.3f} to {self.t_vive[-1]:.3f}")
            logger.info(f"Robot signal range: {np.min(self.x_robot):.3f} to {np.max(self.x_robot):.3f}")
            logger.info(f"Vive signal range: {np.min(self.x_vive):.3f} to {np.max(self.x_vive):.3f}")

            # Calculate latency using cross-correlation
            latency, info = get_latency(
                self.x_vive, self.t_vive,
                self.x_robot, self.t_robot,
                resample_dt=1/100,   # 10ms resampling for better stability
                force_positive=False  # Allow both positive and negative latency
            )

            logger.info(f"Calculated latency: {latency*1000:.2f} ms")
            logger.info(f"Correlation max at lag: {info['lags'][np.argmax(info['correlation'])]*1000:.2f} ms")
            return latency, info

        except Exception as e:
            logger.error(f"Failed to analyze latency: {e}")
            raise

    def plot_results(self, latency: float, info: Dict[str, Any]) -> None:
        """Plot the latency analysis results."""
        try:
            import matplotlib.pyplot as plt

            logger.info("Creating latency analysis plots...")

            # Convert to numpy arrays
            t_robot = np.array(self.t_robot)
            x_robot = np.array(self.x_robot)
            t_vive = np.array(self.t_vive)
            x_vive = np.array(self.x_vive)

            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            fig.suptitle(f'Vive Tracker Latency Analysis (Axis {self.robot_axis} -> {self.vive_axis})', fontsize=16)

            # Plot 1: Cross-correlation
            ax = axes[0]
            ax.plot(info['lags'] * 1000, info['correlation'])
            ax.axvline(x=latency * 1000, color='r', linestyle='--',
                      label=f'Latency: {latency*1000:.2f} ms')
            ax.set_xlabel('Lag [ms]')
            ax.set_ylabel('Cross-correlation')
            ax.set_title('Cross-correlation Analysis')
            ax.legend()
            ax.grid(True)

            # Plot 2: Raw signals
            ax = axes[1]
            ax.plot(t_robot, x_robot, 'b-', label='Robot Actual Position', linewidth=2)
            ax.plot(t_vive, x_vive, 'g-', label='Vive Tracker', linewidth=2)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Position')
            ax.set_title('Raw Signals (Actual Robot Position vs Vive)')
            ax.legend()
            ax.grid(True)

            # Plot 3: Aligned signals
            ax = axes[2]
            t_samples = info['t_samples'] - info['t_samples'][0]
            ax.plot(t_samples, info['x_target'], 'b-', label='Robot Position', linewidth=2)
            ax.plot(t_samples - latency, info['x_actual'], 'g-', label='Vive Tracker (aligned)', linewidth=2)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Normalized Position')
            ax.set_title(f'Aligned Signals (Latency: {latency*1000:.2f} ms)')
            ax.legend()
            ax.grid(True)

            plt.tight_layout()

            # Save plot if configured
            if self.config['output']['plot_save']:
                plot_file = self.config['output']['plot_file']
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {plot_file}")

            # Show plot if configured
            if self.config['output']['plot_results']:
                plt.show()

        except ImportError:
            logger.warning("Matplotlib not available, skipping plotting")
        except Exception as e:
            logger.error(f"Failed to create plots: {e}")

    def save_data(self, latency: float, info: Dict[str, Any]) -> None:
        """Save the collected data and analysis results."""
        try:
            if not self.config['output']['save_data']:
                return

            data_file = self.config['output']['data_file']

            # Prepare data for saving
            save_data = {
                'config': self.config,
                't_robot': np.array(self.t_robot),
                'x_robot': np.array(self.x_robot),
                't_vive': np.array(self.t_vive),
                'x_vive': np.array(self.x_vive),
                'latency': latency,
                'analysis_info': info,
                'robot_axis': self.robot_axis,
                'vive_axis': self.vive_axis,
                'test_frequency': self.config['test']['frequency'],
                'test_duration': self.duration
            }

            np.savez(data_file, **save_data)
            logger.info(f"Data saved to {data_file}")

        except Exception as e:
            logger.error(f"Failed to save data: {e}")

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            logger.info("Cleaning up resources...")

            if self.robot:
                # Move robot back to home pose
                try:
                    if self.robot.is_dual_arm:
                        home_pose = self.robot.home_pose_left
                    else:
                        home_pose = self.robot.home_pose

                    if home_pose is not None:
                        target_pose = [
                            home_pose[0], home_pose[1], home_pose[2],
                            home_pose[6], home_pose[3], home_pose[4], home_pose[5]
                        ]
                        self.robot.robot.SendCartesianMotionForce(target_pose)
                        precise_sleep(2.0)
                except Exception as e:
                    logger.warning(f"Failed to return robot to home: {e}")

                # Clean up robot
                try:
                    if hasattr(self.robot, 'stop_server'):
                        self.robot.stop_server()
                except:
                    pass

            if self.vive_tracker:
                # Clean up Vive Tracker
                try:
                    if hasattr(self.vive_tracker, 'stop_server'):
                        self.vive_tracker.stop_server()
                except:
                    pass

            logger.info("Cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def run(self) -> bool:
        """Run the complete calibration process."""
        try:
            # Initialize devices
            if not self.initialize_devices():
                return False

            # Run calibration test
            if not self.run_calibration():
                return False

            # Analyze latency
            latency, info = self.analyze_latency()

            # Display results
            logger.info("=" * 50)
            logger.info("LATENCY CALIBRATION RESULTS")
            logger.info("=" * 50)
            logger.info(f"Robot motion axis: {self.robot_axis}")
            logger.info(f"Vive Tracker reading axis: {self.vive_axis}")
            logger.info(f"Test frequency: {self.config['test']['frequency']} Hz")
            logger.info(f"Test duration: {self.duration} seconds")
            logger.info(f"Calculated latency: {latency*1000:.2f} ms")
            logger.info(f"Robot state samples: {len(self.t_robot)}")
            logger.info(f"Vive Tracker samples: {len(self.t_vive)}")
            logger.info("=" * 50)

            # Plot results
            self.plot_results(latency, info)

            # Save data
            self.save_data(latency, info)

            return True

        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return False
        finally:
            self.cleanup()


def main():
    """Main function for Vive Tracker latency calibration."""
    parser = argparse.ArgumentParser(description="Vive Tracker Latency Calibration")
    parser.add_argument("--config", "-c", type=str,
                       default="config_vive_tracker_latency.yaml",
                       help="Configuration file path")
    parser.add_argument("--robot-sn", type=str, default=None,
                       help="Override robot serial number from config")
    parser.add_argument("--frequency", "-f", type=float, default=None,
                       help="Override test frequency from config")
    parser.add_argument("--duration", "-d", type=float, default=None,
                       help="Override test duration from config")
    parser.add_argument("--robot-axis", type=int, default=None,
                       help="Override robot motion axis from config")
    parser.add_argument("--vive-axis", type=int, default=None,
                       help="Override Vive Tracker reading axis from config")

    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    # Create calibrator
    calibrator = ViveTrackerLatencyCalibrator(args.config)

    # Override config values if provided
    if args.robot_sn:
        calibrator.config['robot']['robot_sn'] = args.robot_sn
    if args.frequency:
        calibrator.config['test']['frequency'] = args.frequency
        calibrator.dt = 1.0 / args.frequency
    if args.duration:
        calibrator.config['test']['duration'] = args.duration
        calibrator.duration = args.duration
    if args.robot_axis is not None:
        calibrator.config['test']['robot_axis'] = args.robot_axis
        calibrator.robot_axis = args.robot_axis
    if args.vive_axis is not None:
        calibrator.config['test']['vive_axis'] = args.vive_axis
        calibrator.vive_axis = args.vive_axis

    try:
        # Run calibration
        success = calibrator.run()
        if success:
            logger.info("Latency calibration completed successfully")
            sys.exit(0)
        else:
            logger.error("Latency calibration failed")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Calibration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
