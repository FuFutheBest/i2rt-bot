#!/usr/bin/env python3
"""
Real-time Action Visualizer - visualizes action dimensions over time using plotly offline mode.
Generates HTML files that can be viewed in a browser, updated in background thread without blocking main thread.

Author: Assistant
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict, deque
from typing import Dict, Optional, List, Tuple, Union
import threading
import time
import os
from pathlib import Path
from utils.logger_config import logger


class ActionVisualizer:
    """
    Real-time visualizer for action dimensions using plotly offline mode.
    Generates HTML files that can be viewed in a browser.
    All updates happen in background thread, non-blocking.
    """

    def __init__(
            self,
            max_history: int = 500,
            update_interval_ms: int = 50,
            figsize: tuple = (12, 8),
            window_size: int = 100,
            enable: bool = True,
            output_dir: str = "./action_visualizations",
            html_filename: str = "action_visualization.html",
            time_range: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize the action visualizer.

        Args:
            max_history: Maximum number of time steps to keep in history
            update_interval_ms: Update interval in milliseconds for animation
            figsize: Figure size (width, height) - used for plotly layout
            window_size: Display window size (number of latest steps to show).
                        If -1, show all history. Default is 100.
            enable: Whether to enable visualization
            output_dir: Directory to save HTML visualization files
            html_filename: Name of the HTML file to generate
            time_range: Optional tuple (start_step, end_step) to control x-axis display range.
                       If None, uses window_size. If provided, only shows data within this range.
                       Examples: (0, 100) shows first 100 steps, (50, 150) shows steps 50-150.
        """
        self.enable = enable
        if not self.enable:
            logger.info("ActionVisualizer is disabled")
            return

        self.max_history = max_history
        self.update_interval_ms = update_interval_ms
        self.figsize = figsize
        self.window_size = window_size if window_size != -1 else None  # None means show all
        self.time_range = time_range  # Optional (start_step, end_step) for x-axis range

        # Output configuration
        self.output_dir = Path(output_dir)
        self.html_filename = html_filename
        self.html_path = self.output_dir / self.html_filename

        # Create output directory if it doesn't exist
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create output directory {self.output_dir}: {e}")

        # Data storage: {device_name: deque of action arrays}
        self.action_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.time_history: deque = deque(maxlen=max_history)

        # Track time
        self.step_counter = 0

        # Device and dimension info
        self.device_names: List[str] = []
        self.action_dims: Dict[str, int] = {}  # {device_name: action_dimension}

        # Thread safety
        self.lock = threading.Lock()

        # Flag to track if figure is initialized
        self.initialized = False

        # Background update thread
        self.update_thread: Optional[threading.Thread] = None
        self.running = False
        self.update_interval_sec = update_interval_ms / 1000.0

        window_info = "all data" if self.window_size is None else f"window_size={self.window_size}"
        time_range_info = f", time_range={self.time_range}" if self.time_range else ""
        logger.info(
            f"ActionVisualizer initialized with max_history={max_history}, display={window_info}{time_range_info}")
        logger.info(f"HTML visualization will be saved to: {self.html_path}")
        logger.info(f"Open this file in your browser and refresh periodically to see updates")

    def add_action(self, action_dict: Dict[str, np.ndarray]) -> None:
        """
        Add a new action to the visualization.
        This is a non-blocking operation that only adds data to history.

        Args:
            action_dict: Dictionary mapping device names to action arrays
        """
        if not self.enable:
            return

        # Fast path: minimize lock time by only locking for data access
        # Initialize on first call (this needs lock)
        if not self.initialized:
            with self.lock:
                # Double-check pattern to avoid race condition
                if not self.initialized:
                    self._initialize(action_dict)
                    self.initialized = True
                    # Start background update thread
                    self._start_update_thread()

        # Add data with minimal lock time
        with self.lock:
            # Add action to history (fast operation)
            for device_name, action in action_dict.items():
                self.action_history[device_name].append(action.copy())

            # Add time step
            self.time_history.append(self.step_counter)
            self.step_counter += 1

    def _initialize(self, action_dict: Dict[str, np.ndarray]) -> None:
        """
        Initialize the visualizer based on action dimensions.

        Args:
            action_dict: Dictionary mapping device names to action arrays
        """
        # Get device names and dimensions
        self.device_names = sorted(action_dict.keys())
        for device_name, action in action_dict.items():
            self.action_dims[device_name] = len(action)

        # Calculate total number of dimensions
        total_dims = sum(self.action_dims.values())

        logger.info(f"Initializing ActionVisualizer with {len(self.device_names)} devices, "
                    f"total {total_dims} dimensions")

    def _start_update_thread(self) -> None:
        """Start background thread for periodic visualization updates."""
        if self.update_thread is not None:
            return

        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        logger.info("ActionVisualizer update thread started")

    def _update_loop(self) -> None:
        """Background thread loop for periodic visualization updates."""
        while self.running:
            try:
                self._update_visualization()
                time.sleep(self.update_interval_sec)
            except Exception as e:
                logger.warning(f"Error in visualization update loop: {e}")
                time.sleep(self.update_interval_sec)

    def _update_visualization(self) -> None:
        """
        Internal update method that performs the actual visualization update.
        This is called from the background thread.
        Plotly operations are thread-safe and can be called from any thread.
        """
        if not self.enable or not self.initialized:
            return

        # Get data snapshot with minimal lock time
        with self.lock:
            if len(self.time_history) == 0:
                return

            # Create copies of data for visualization (release lock quickly)
            time_array = np.array(self.time_history)
            action_history_snapshot = {}
            for device_name, history in self.action_history.items():
                action_history_snapshot[device_name] = np.array(history)

        # Apply time_range filter if specified (takes priority over window_size)
        if self.time_range is not None:
            start_step, end_step = self.time_range
            # Find indices within the time range
            mask = (time_array >= start_step) & (time_array <= end_step)
            if np.any(mask):
                time_array = time_array[mask]
                # Apply same mask to all action arrays
                for device_name in action_history_snapshot:
                    action_history_snapshot[device_name] = action_history_snapshot[device_name][mask]
            else:
                # No data in range, skip this update
                logger.debug(f"No data in time_range {self.time_range}, skipping update")
                return
        # Otherwise, apply window_size filter if specified
        elif self.window_size is not None and len(time_array) > self.window_size:
            # Take the last window_size steps
            time_array = time_array[-self.window_size:]
            for device_name in action_history_snapshot:
                action_history_snapshot[device_name] = action_history_snapshot[device_name][-self.window_size:]

        # Calculate total number of dimensions
        total_dims = sum(self.action_dims.values())

        if total_dims == 0:
            return

        # Create subplots
        n_cols = min(3, total_dims)  # Max 3 columns
        n_rows = (total_dims + n_cols - 1) // n_cols

        # Create figure with subplots
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f'{device_name}[{i}]'
                            for device_name in self.device_names
                            for i in range(self.action_dims[device_name])],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )

        # Update each subplot
        dim_idx = 0
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for device_name in self.device_names:
            if device_name not in action_history_snapshot:
                continue

            action_array = action_history_snapshot[device_name]

            if len(action_array) == 0:
                continue

            # Ensure action_array is 2D
            if action_array.ndim == 1:
                action_array = action_array.reshape(-1, 1)

            # Ensure time_array matches action_array length
            # (time_range or window_size filtering already applied above)
            current_time_array = time_array
            if len(current_time_array) != len(action_array):
                # This shouldn't happen if filtering is correct, but handle it gracefully
                min_len = min(len(current_time_array), len(action_array))
                current_time_array = current_time_array[:min_len]
                action_array = action_array[:min_len]

            n_dims = action_array.shape[1]

            for i in range(n_dims):
                if dim_idx >= total_dims:
                    break

                row = (dim_idx // n_cols) + 1
                col = (dim_idx % n_cols) + 1

                color = colors[dim_idx % len(colors)]

                # Add trace
                fig.add_trace(
                    go.Scatter(
                        x=current_time_array,
                        y=action_array[:, i],
                        mode='lines',
                        name=f'{device_name}[{i}]',
                        line=dict(color=color, width=1.5),
                        showlegend=False
                    ),
                    row=row,
                    col=col
                )

                # Update axes labels
                fig.update_xaxes(title_text="Time Step", row=row, col=col)
                fig.update_yaxes(title_text="Action Value", row=row, col=col)

                dim_idx += 1

        # Update layout
        fig.update_layout(
            title_text="Real-time Action Visualization",
            title_x=0.5,
            height=self.figsize[1] * 100,  # Convert to pixels (approximate)
            width=self.figsize[0] * 100,
            template="plotly_white"
        )

        # Save to HTML file (plotly offline mode is thread-safe)
        try:
            fig.write_html(
                str(self.html_path),
                config={'displayModeBar': True, 'displaylogo': False},
                auto_open=False
            )
        except Exception as e:
            logger.warning(f"Error saving visualization HTML: {e}")

    def update(self) -> None:
        """
        Update the plot with current data.
        DEPRECATED: Updates now happen automatically in background thread.
        This method is kept for backward compatibility but does nothing.
        """
        # Visualization updates are now handled by background thread
        # This method is kept for backward compatibility
        pass

    def close(self) -> None:
        """Stop background thread and cleanup."""
        if not self.enable:
            return

        # Stop background update thread
        self.running = False
        if self.update_thread is not None:
            self.update_thread.join(timeout=1.0)
            self.update_thread = None

        logger.info(f"ActionVisualizer closed. Final visualization saved to: {self.html_path}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()