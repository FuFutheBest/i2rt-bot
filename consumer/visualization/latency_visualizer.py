#!/usr/bin/env python3
"""
Latency Consumer - Device latency monitoring and visualization.

This module provides latency monitoring functionality by consuming device manager data
and calculating latencies between devices.

Author: Jun Lv
"""

import time
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from typing import Any, Dict, List

try:
    from .base import BaseConsumer
except ImportError:
    from base import BaseConsumer

from utils.logger_config import logger

# Set matplotlib backend
import matplotlib


class LatencyConsumer(BaseConsumer):
    """
    Latency consumer for monitoring device latencies.
    
    This consumer calculates and visualizes:
    - Individual device latencies
    - Inter-device synchronization latencies
    - Real-time latency trends
    """
    
    def __init__(self, summary_shm_name: str = "device_summary_data", 
                 history_size: int = 100, smoothing_window: int = 5) -> None:
        """
        Initialize the latency consumer.
        
        Args:
            summary_shm_name: Name of the summary shared memory
            history_size: Size of latency history buffer
            smoothing_window: Window size for latency smoothing
        """
        super().__init__(summary_shm_name)
        
        matplotlib.use('TkAgg')
        self.history_size = history_size
        self.smoothing_window = smoothing_window
        
        # Latency tracking
        self.device_history: Dict[str, deque] = {}
        self.latency_history: Dict[str, deque] = {}
        
        # Visualization
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.ani = None
    
    def _parse_summary_header(self) -> None:
        """Parse the summary SHM header and initialize latency tracking."""
        super()._parse_summary_header()
        
        # Initialize history storage for each device
        for device_key in self.devices:
            self.device_history[device_key] = deque(maxlen=self.history_size)
        
        # Initialize latency history for device pairs
        device_keys = list(self.devices.keys())
        for i in range(len(device_keys)):
            for j in range(i + 1, len(device_keys)):
                dev1_name = self.devices[device_keys[i]]['name']
                dev2_name = self.devices[device_keys[j]]['name']
                pair_name = f"{dev1_name}-{dev2_name}"
                self.latency_history[pair_name] = deque(maxlen=self.history_size)
                logger.info(f"Pair {pair_name} initialized")
    
    def _calculate_latencies(self) -> None:
        """Calculate latencies between device pairs."""
        current_time_ns = time.time_ns()
        
        # Update device timestamps
        for device_key in self.devices:
            data = self._read_device_data(device_key)
            if data:
                timestamp_ns, _ = data
                
                # Apply hardware latency compensation
                # Convert hardware_latency_ms to nanoseconds and subtract from server timestamp
                hardware_latency_ns = int(self.devices[device_key]['hardware_latency_ms'] * 1e6)
                real_timestamp_ns = timestamp_ns - hardware_latency_ns
                
                self.devices[device_key]['last_timestamp'] = real_timestamp_ns
                self.devices[device_key]['last_update_time'] = current_time_ns
                
                # Store in history (use real timestamp for latency calculation)
                self.device_history[device_key].append({
                    'timestamp': real_timestamp_ns,
                    'current_time': current_time_ns,
                    'latency': (current_time_ns - real_timestamp_ns) / 1e6  # Convert to ms
                })
        
        # Calculate latencies between device pairs
        device_keys = list(self.devices.keys())
        for i in range(len(device_keys)):
            for j in range(i + 1, len(device_keys)):
                dev1_key = device_keys[i]
                dev2_key = device_keys[j]
                dev1_name = self.devices[dev1_key]['name']
                dev2_name = self.devices[dev2_key]['name']
                pair_name = f"{dev1_name}-{dev2_name}"
                
                dev1_info = self.devices[dev1_key]
                dev2_info = self.devices[dev2_key]
                
                if (dev1_info['last_timestamp'] is not None and 
                    dev2_info['last_timestamp'] is not None):
                    
                    # Calculate absolute time difference
                    time_diff = abs(dev1_info['last_timestamp'] - dev2_info['last_timestamp'])
                    time_diff_ms = time_diff / 1e6  # Convert to ms
                    
                    self.latency_history[pair_name].append({
                        'timestamp': current_time_ns,
                        'latency': time_diff_ms,
                        'dev1_timestamp': dev1_info['last_timestamp'],
                        'dev2_timestamp': dev2_info['last_timestamp']
                    })
    
    def _smooth_value(self, values: deque) -> float:
        """Calculate smoothed value using moving average."""
        if len(values) == 0:
            return 0.0
        
        if len(values) < self.smoothing_window:
            return sum(values) / len(values)
        
        recent_values = list(values)[-self.smoothing_window:]
        return sum(recent_values) / len(recent_values)
    
    def _update_plot(self, frame: Any) -> List:
        """Update the latency visualization plot."""
        if not self.running:
            return
        
        try:
            # Calculate latencies
            self._calculate_latencies()
            
            # Clear previous plots
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            
            # Plot 1: Individual device latencies
            device_keys = list(self.devices.keys())
            device_latencies = []
            device_names = []
            
            for device_key in device_keys:
                device_names.append(self.devices[device_key]['name'])
                if self.device_history[device_key]:
                    avg_latency = self._smooth_value(
                        deque([h['latency'] for h in self.device_history[device_key]], maxlen=self.smoothing_window)
                    )
                    device_latencies.append(avg_latency)
                else:
                    # Show 0 latency for devices without data
                    device_latencies.append(0.0)
            
            if device_latencies:
                bars1 = self.ax1.bar(device_names, device_latencies, color='skyblue', alpha=0.7)
                self.ax1.set_title('Individual Device Latencies (ms)')
                self.ax1.set_ylabel('Latency (ms)')
                self.ax1.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars1, device_latencies):
                    self.ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                f'{value:.2f}', ha='center', va='bottom', fontsize=8)
            
            # Plot 2: Inter-device latencies
            pair_names = []
            pair_latencies = []
            
            for pair_name, history in self.latency_history.items():
                if history:
                    avg_latency = self._smooth_value(
                        deque([h['latency'] for h in history], maxlen=self.smoothing_window)
                    )
                    pair_latencies.append(avg_latency)
                    pair_names.append(pair_name)
            
            if pair_latencies:
                bars2 = self.ax2.bar(pair_names, pair_latencies, color='lightcoral', alpha=0.7)
                self.ax2.set_title('Inter-Device Latency Differences (ms)')
                self.ax2.set_ylabel('Latency Difference (ms)')
                self.ax2.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars2, pair_latencies):
                    self.ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                f'{value:.2f}', ha='center', va='bottom', fontsize=8)
            
            # Plot 3: Real-time latency trends
            if self.latency_history:
                # Get the most recent data for trend plotting
                for pair_name, history in self.latency_history.items():
                    if len(history) > 1:
                        timestamps = [(h['timestamp'] - history[0]['timestamp']) / 1e9 for h in history]
                        latencies = [h['latency'] for h in history]
                        self.ax3.plot(timestamps, latencies, label=pair_name, alpha=0.7)
                
                self.ax3.set_title('Real-time Latency Trends')
                self.ax3.set_xlabel('Time (s)')
                self.ax3.set_ylabel('Latency (ms)')
                self.ax3.grid(True, alpha=0.3)
                self.ax3.legend()
            
            # Update status text
            current_time = time.strftime('%H:%M:%S')
            status_text = f"Last Update: {current_time}\n"
            status_text += f"Connected Devices: {len([d for d in self.devices.values() if d['last_timestamp'] is not None])}/{len(self.devices)}\n"
            
            if self.latency_history:
                max_latency = max([max([h['latency'] for h in history]) for history in self.latency_history.values() if history])
                status_text += f"Max Latency: {max_latency:.2f} ms"
            
            self.ax3.text(0.02, 0.98, status_text, transform=self.ax3.transAxes, 
                         verticalalignment='top', fontsize=8, 
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
        except Exception as e:
            logger.error(f"Error updating plot: {e}")
    
    def start_visualization(self, update_interval: int = 100) -> None:
        """
        Start the latency visualization.
        
        Args:
            update_interval: Update interval in milliseconds
        """
        if not self.summary_shm:
            logger.error("Not connected to summary SHM")
            return
        
        logger.info("Starting latency visualization")
        self.running = True
        
        # Configure matplotlib for Chinese text support
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Create figure and subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle('Device Latency Monitor', fontsize=16)
        
        # Setup animation
        self.ani = animation.FuncAnimation(
            self.fig, self._update_plot, interval=update_interval, 
            blit=False, cache_frame_data=False
        )
        
        try:
            plt.show()
        except KeyboardInterrupt:
            logger.info("Visualization interrupted")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the latency consumer."""
        if not self.running:
            return
        
        logger.info("Stopping latency consumer")
        self.running = False
        
        if hasattr(self, 'ani') and self.ani is not None and hasattr(self.ani, 'event_source') and self.ani.event_source is not None:
            self.ani.event_source.stop()
        
        if hasattr(self, 'fig'):
            plt.close(self.fig)
        
        # Call parent stop method
        super().stop()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of the latency consumer."""
        status = super().get_status()
        
        # Add latency-specific information
        status['history_size'] = self.history_size
        status['smoothing_window'] = self.smoothing_window
        status['device_pairs'] = len(self.latency_history)
        
        # Add device details with latency information
        device_details = {}
        for device_key, device_info in self.devices.items():
            device_details[device_key] = {
                'type': device_info['type'],
                'id': device_info['id'],
                'last_timestamp': device_info['last_timestamp'],
                'fps': device_info['fps'],
                'hardware_latency_ms': device_info['hardware_latency_ms'],
                'history_samples': len(self.device_history[device_key])
            }
        status['device_details'] = device_details
        
        return status


def main() -> None:
    """Main function to run the latency consumer."""
    parser = argparse.ArgumentParser(description="Latency Consumer - Device Latency Monitor")
    parser.add_argument("--summary-shm", "-s", default="device_summary_data",
                        help="Summary shared memory name (default: device_summary_data)")
    parser.add_argument("--history-size", type=int, default=100,
                        help="History size for latency tracking (default: 100)")
    parser.add_argument("--smoothing", "-w", type=int, default=5,
                        help="Smoothing window size (default: 5)")
    parser.add_argument("--update-interval", "-i", type=int, default=100,
                        help="Update interval in milliseconds (default: 100)")
    parser.add_argument("--status", action="store_true",
                        help="Show consumer status and exit")
    
    args = parser.parse_args()
    
    # Create latency consumer
    consumer = LatencyConsumer(
        summary_shm_name=args.summary_shm,
        history_size=args.history_size,
        smoothing_window=args.smoothing
    )
    
    if args.status:
        # Show status
        status = consumer.get_status()
        logger.info("Latency Consumer Status:")
        logger.info(f"Connected: {status['connected']}")
        logger.info(f"Running: {status['running']}")
        logger.info(f"Devices: {status['devices']}")
        logger.info(f"Device pairs: {status['device_pairs']}")
        logger.info(f"Summary SHM: {status['summary_shm_name']}")
        logger.info(f"History size: {status['history_size']}")
        logger.info(f"Smoothing window: {status['smoothing_window']}")
        
        if status['device_details']:
            logger.info("\nDevice Details:")
            for device_key, details in status['device_details'].items():
                logger.info(f"  Device {device_key} (ID: {details['id']}): "
                           f"Last timestamp: {details['last_timestamp']}, "
                           f"Samples: {details['history_samples']}")
        return
    
    logger.info("Latency Consumer - Device Latency Monitor")
    logger.info("=========================================")
    logger.info(f"Summary SHM: {args.summary_shm}")
    logger.info(f"History size: {args.history_size}")
    logger.info(f"Smoothing window: {args.smoothing}")
    logger.info(f"Update interval: {args.update_interval}ms")
    logger.info("")
    
    # Try to connect
    if not consumer.connect():
        logger.error("Failed to connect to summary SHM. Make sure the Device Manager is running.")
        return
    
    try:
        logger.info("Latency monitor is running. Press Ctrl+C to stop...")
        consumer.start_visualization(update_interval=args.update_interval)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        consumer.stop()


if __name__ == "__main__":
    main() 