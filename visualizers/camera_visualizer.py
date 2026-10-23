#!/usr/bin/env python3
"""
Camera Visualizer - Real-time visualization for OpenCV camera streams.

This visualizer provides real-time camera stream display with floating
status information overlay.

Author: Jun Lv
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import argparse
import sys
import os
from typing import List, Any, Optional, Tuple, Dict
import cv2

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .base_visualizer import BaseVisualizer
except ImportError:
    from visualizers.base_visualizer import BaseVisualizer

from utils.logger_config import logger


class CameraVisualizer(BaseVisualizer):
    """
    Camera visualizer to display real-time camera frames from OpenCVCameraDevice shared memory.
    Supports both grayscale and color camera streams with status overlay.
    """

    def __init__(self, shared_memory_name: str = "opencv_camera_0_data", data_dtype: Any = np.uint8,
                 smoothing_window: int = 10, overlay_image: Optional[np.ndarray] = None,
                 overlay_alpha: float = 0.5, device_info: Dict[str, Any] = None,
                 enable_overlay: bool = False) -> None:
        """
        Initialize the camera visualizer.

        Args:
            shared_memory_name: Name of the shared memory to read from
            data_dtype: Expected data type of the numpy array
            smoothing_window: Number of samples for FPS and latency smoothing
            overlay_image: Optional image to overlay on top of camera stream (numpy array)
            overlay_alpha: Transparency of overlay image (0.0 to 1.0, default: 0.5)
            device_info: Optional dictionary containing device information.
                        If provided and contains 'device_class_id', will use it to select overlay image:
                        device_class_id=0 -> left.jpg, device_class_id=1 -> right.jpg
            enable_overlay: Boolean flag to enable/disable overlay image blending (default: False)
        """
        # Initialize base visualizer
        super().__init__(shared_memory_name, data_dtype)

        # Override matplotlib setup for camera visualization
        plt.close(self.fig)  # Close the default status monitoring figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.ax.set_title("Camera Stream")
        self.im = None
        self.info_text = None
        self.current_shape: Optional[Tuple[int, int]] = None

        # Load hardcoded overlay image if not provided
        if overlay_image is None:
            device_class_id = None
            if device_info is not None and 'device_class_id' in device_info:
                device_class_id = device_info['device_class_id']
            overlay_image = self._load_hardcoded_overlay(device_class_id=device_class_id)

        self.overlay_image: Optional[np.ndarray] = overlay_image
        self.overlay_alpha: float = overlay_alpha
        self.enable_overlay: bool = enable_overlay

        # Auto-save: save images per camera (image only, no text overlay)
        self._device_class_id: int = (device_info or {}).get("device_class_id", 0)
        self._auto_save_count: int = 0
        self._auto_save_target: int = 0
        self._auto_save_dir: str = "data/UMI_new"

    def _load_hardcoded_overlay(self, device_class_id: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Load overlay image from calibration/hikcamera/reference_images/left.jpg or right.jpg.

        Args:
            device_class_id: Device ID to determine which image to load.
                      If device_class_id is 0, loads left_robot_129.jpg.
                      If device_class_id is 1, loads right_robot_129.jpg.
                      If None, defaults to left_robot_129.jpg.

        Returns:
            Overlay image as numpy array (RGB), or None if file not found
        """
        pictures_dir = os.path.expanduser("calibration/hikcamera/reference_images")
        if device_class_id == 0:
            image_name = "left_robot_129.jpg"
        elif device_class_id == 1:
            image_name = "right_robot_129.jpg"
        else:
            image_name = "left_robot_129.jpg"

        image_path = os.path.join(pictures_dir, image_name)
        if not os.path.exists(image_path):
            logger.warning(f"Overlay image not found at {image_path}")
            return None

        try:
            overlay = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if overlay is not None:
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                logger.info(f"Loaded overlay image from {image_path} (shape: {overlay.shape})")
                return overlay
            overlay = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if overlay is not None:
                overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
                logger.info(f"Loaded overlay image (grayscale) from {image_path} (shape: {overlay.shape})")
                return overlay
        except Exception as exc:
            logger.warning(f"Failed to load overlay image {image_path}: {exc}")
        return None

    def _ensure_figure_matches_resolution(self, frame_shape: Tuple[int, int]) -> None:
        """Resize figure window to match incoming frame resolution to avoid distortion."""
        if self.current_shape == frame_shape:
            return

        self.current_shape = frame_shape
        frame_height, frame_width = frame_shape
        dpi = self.fig.get_dpi() or plt.rcParams.get("figure.dpi", 100)

        min_size_inches = 2.0
        fig_width = max(frame_width / dpi, min_size_inches)
        fig_height = max(frame_height / dpi, min_size_inches)

        self.fig.set_size_inches(fig_width, fig_height, forward=True)
        self.ax.set_aspect('equal', adjustable='box')

    def set_overlay_image(self, overlay_image: Optional[np.ndarray], overlay_alpha: float = 0.5) -> None:
        """
        Set or update the overlay image.

        Args:
            overlay_image: Image to overlay (numpy array) or None to remove overlay
            overlay_alpha: Transparency of overlay image (0.0 to 1.0)
        """
        self.overlay_image = overlay_image
        self.overlay_alpha = overlay_alpha
        logger.info(f"Overlay image updated. Alpha: {overlay_alpha}")

    def set_enable_overlay(self, enable: bool) -> None:
        """
        Enable or disable overlay image blending.

        Args:
            enable: Boolean flag to enable (True) or disable (False) overlay blending
        """
        self.enable_overlay = enable
        logger.info(f"Overlay blending {'enabled' if enable else 'disabled'}")

    def _blend_images(self, base_image: np.ndarray, overlay_image: np.ndarray, alpha: float) -> np.ndarray:
        """
        Blend base image with overlay image.

        Args:
            base_image: Base camera image
            overlay_image: Overlay image to blend on top
            alpha: Transparency of overlay (0.0 to 1.0)

        Returns:
            Blended image
        """
        # Ensure both images are in the same format
        # Convert to float for blending
        base = base_image.astype(np.float32)
        overlay = overlay_image.astype(np.float32)

        # Normalize to [0, 1] range if needed
        if base.max() > 1.0:
            base = base / 255.0
        if overlay.max() > 1.0:
            overlay = overlay / 255.0

        # Resize overlay to match base image if needed
        if overlay.shape[:2] != base.shape[:2]:
            target_size = (base.shape[1], base.shape[0])  # cv2.resize uses (width, height)
            overlay = cv2.resize(overlay, target_size, interpolation=cv2.INTER_LINEAR)

        # Handle grayscale vs color
        if len(base.shape) == 2 and len(overlay.shape) == 3:
            # Base is grayscale, overlay is color - convert overlay to grayscale
            overlay = np.mean(overlay, axis=2, keepdims=False)
            overlay = np.expand_dims(overlay, axis=2) if len(base.shape) == 2 else overlay
        elif len(base.shape) == 3 and len(overlay.shape) == 2:
            # Base is color, overlay is grayscale - convert to 3 channels
            overlay = np.stack([overlay] * base.shape[2], axis=2)
        elif len(base.shape) == 2 and len(overlay.shape) == 2:
            # Both grayscale
            pass
        elif len(base.shape) == 3 and len(overlay.shape) == 3:
            # Both color - ensure same number of channels
            if base.shape[2] != overlay.shape[2]:
                if overlay.shape[2] == 1:
                    overlay = np.repeat(overlay, base.shape[2], axis=2)
                elif base.shape[2] == 1:
                    base = np.repeat(base, overlay.shape[2], axis=2)
                else:
                    # Take first 3 channels if overlay has more
                    overlay = overlay[:, :, :base.shape[2]]

        # Blend: result = base * (1 - alpha) + overlay * alpha
        blended = base * (1.0 - alpha) + overlay * alpha

        # Convert back to uint8
        blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

        return blended

    def update_plot(self, frame: Any) -> List:
        """Update the matplotlib plot with new data."""
        timestamp_ns, data_array = self.read_latest_frame()

        if timestamp_ns is None or data_array is None:
            return []

        # Calculate FPS
        self.calculate_fps(timestamp_ns)

        # Convert timestamp to readable format
        timestamp_s = timestamp_ns / 1e9
        dt = datetime.fromtimestamp(timestamp_s)
        time_str = dt.strftime("%H:%M:%S.%f")[:-3]  # millisecond precision

        # Calculate latency (difference between current time and frame timestamp)
        latency_ms = self.calculate_latency(timestamp_ns)

        # Handle color vs grayscale images
        if len(data_array.shape) == 3 and data_array.shape[2] == 3:
            # Color image
            cmap = None
            base_image = data_array.copy()
            shape_str = f"{data_array.shape[1]}x{data_array.shape[0]}x{data_array.shape[2]}"
        else:
            # Grayscale image
            cmap = 'gray'
            base_image = data_array.copy()
            shape_str = f"{data_array.shape[1]}x{data_array.shape[0]}"

        # Blend with overlay image if provided and enabled (_blend_images resizes overlay to match base)
        display_data = base_image.copy()
        if self.enable_overlay and self.overlay_image is not None:
            try:
                display_data = self._blend_images(display_data, self.overlay_image, self.overlay_alpha)
            except Exception as e:
                logger.warning(f"Failed to blend overlay image: {e}")

        # Auto-save: save display image per camera (image only, no text overlay)
        if self._auto_save_count < self._auto_save_target and frame in (0, 60):
            os.makedirs(self._auto_save_dir, exist_ok=True)
            side = "left" if self._device_class_id == 0 else "right"
            idx = self._auto_save_count + 1

            def _save_image(img: np.ndarray, filepath: str) -> bool:
                if img is None or img.size == 0:
                    return False
                out = img if img.max() <= 255 else (img.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
                if len(out.shape) == 3 and out.shape[2] == 3:
                    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                return cv2.imwrite(filepath, out)

            path_display = os.path.join(self._auto_save_dir, f"{side}_{idx}.jpg")
            if _save_image(display_data, path_display):
                logger.info(f"Auto-saved camera image: {path_display}")
            self._auto_save_count += 1

        self._ensure_figure_matches_resolution(data_array.shape[:2])

        if self.im is None:
            # First time: create image plot
            self.ax.clear()
            self.im = self.ax.imshow(display_data, cmap=cmap, aspect='equal')
            self.ax.set_title("Camera Stream")
            self.ax.axis('off')  # Hide axes for cleaner image view
        else:
            # Update existing image
            self.im.set_array(display_data)
            if cmap == 'gray':  # Only set clim for grayscale
                self.im.set_clim(vmin=np.min(display_data), vmax=np.max(display_data))

        # Remove old info text if exists
        if self.info_text is not None:
            self.info_text.remove()

        info_text = f"Size: {shape_str}\nFPS: {self.fps_estimate:.1f}\nTime: {time_str}\nLatency: {latency_ms:.1f}ms"
        # Add floating info box in top-left corner
        self.info_text = self.ax.text(0.02, 0.98, info_text,
                                     transform=self.ax.transAxes,
                                     verticalalignment='top',
                                     horizontalalignment='left',
                                     fontsize=10,
                                     fontfamily='monospace',
                                     bbox=dict(boxstyle="round,pad=0.3",
                                              facecolor='black',
                                              alpha=0.7,
                                              edgecolor='white'),
                                     color='white')

        return []

    def start_visualization(self, update_interval: int = 50) -> None:
        """
        Start real-time visualization.

        Args:
            update_interval: Update interval in milliseconds
        """
        if not self.connect():
            return

        logger.info("Starting camera visualization. Close the plot window to stop.")
        logger.info(f"Monitoring: {self.shared_memory_name}")
        logger.info("-" * 50)

        # Set up the animation
        ani = animation.FuncAnimation(self.fig, self.update_plot, interval=update_interval,
                                    blit=False, cache_frame_data=False)

        # Show the plot
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        plt.show()

        # Clean up
        self.shared_memory.close()


def main() -> None:
    """Main function to run the camera visualizer."""
    parser = argparse.ArgumentParser(description="Camera visualization client")
    parser.add_argument("--shared-memory", default="opencv_camera_0_data",
                        help="Shared memory name (default: opencv_camera_0_data)")
    parser.add_argument("--dtype", "-d", default="uint8",
                        choices=['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
                                'float32', 'float64'],
                        help="Data type (default: uint8)")
    parser.add_argument("--interval", "-i", type=int, default=50,
                        help="Update interval in ms (default: 50)")
    parser.add_argument("--smoothing", "-s", type=int, default=5,
                        help="Smoothing window size for FPS and latency (default: 5)")

    args = parser.parse_args()

    # Import common utilities
    from utils.shm_utils import get_dtype

    logger.info(f"Starting Camera visualization...")
    logger.info(f"Shared memory: {args.shared_memory}")
    logger.info(f"Data type: {args.dtype}")

    visualizer = CameraVisualizer(
        shared_memory_name=args.shared_memory,
        data_dtype=get_dtype(args.dtype),
        smoothing_window=args.smoothing
    )

    visualizer.start_visualization(update_interval=args.interval)


if __name__ == "__main__":
    main()
