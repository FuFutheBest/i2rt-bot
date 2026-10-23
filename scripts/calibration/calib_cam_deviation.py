import sys
import os
import time
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger_config import logger
from devices.camera import HikCameraDevice


def detect_and_draw_circles_static(image: np.ndarray, draw_circles: bool = True) -> Tuple[np.ndarray, Optional[Tuple[int, int, int]]]:
    """
    Detect circles in static reference image and optionally draw them in blue for overlay.
    Similar to _detect_and_draw_circles but draws in blue color and doesn't use smoothing.

    Args:
        image: Input static image (numpy array)
        draw_circles: Whether to draw circles on the image (default: True)

    Returns:
        Tuple of (image with detected circles drawn in blue if draw_circles=True, detected circle (x, y, r) or None)
    """
    # Create a copy to avoid modifying the original
    result_image = image.copy()

    # Get image dimensions
    if len(image.shape) == 3:
        height, width = image.shape[:2]
        # Convert to grayscale for circle detection
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        height, width = image.shape
        gray = image

    # Step 1: Extract edges using both Canny and threshold methods
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Method 1: Canny edge detection (more sensitive to edges)
    edges_canny = cv2.Canny(blurred, 30, 100)

    # Method 2: Adaptive threshold (better for varying lighting)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Combine both methods
    edges = cv2.bitwise_or(edges_canny, thresh)

    # Apply morphological operations to connect broken edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 2: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate size thresholds
    min_radius = int(min(height, width) * 0.465)
    max_radius = int(max(height, width) * 0.495)
    min_area = np.pi * (min_radius ** 2) * 0.3
    min_perimeter = 2 * np.pi * min_radius * 0.3

    best_circularity = 0.0
    circularity_threshold = 0.5
    detected_circle = None  # (x, y, r)

    # Step 3-6: Process each contour
    for contour in contours:
        # Step 3: Filter out small noise points
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Approximate contour to reduce noise
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        perimeter = cv2.arcLength(approx, True)
        if perimeter < min_perimeter:
            continue

        # Step 4: Use minimum enclosing circle fitting
        (x, y), radius = cv2.minEnclosingCircle(contour)
        x, y, radius = int(x), int(y), int(radius)

        # Filter by radius range
        if radius < min_radius or radius > max_radius:
            continue

        # Step 5: Calculate circularity
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
        else:
            circularity = 0.0

        # Compare area with enclosing circle area
        enclosing_circle_area = np.pi * (radius ** 2)
        if enclosing_circle_area > 0:
            area_ratio = area / enclosing_circle_area
        else:
            area_ratio = 0.0

        # Combined circularity score
        combined_circularity = 0.6 * circularity + 0.4 * area_ratio

        # Step 6: Only consider high circularity contours as circles
        if combined_circularity >= circularity_threshold:
            # Draw detected circle in blue (RGB: 0, 0, 255) for static reference image
            if draw_circles:
                if len(result_image.shape) == 3:
                    # Color image - draw in blue
                    cv2.circle(result_image, (x, y), radius, (0, 0, 255), 1)
                else:
                    # Grayscale image - draw in white (255)
                    cv2.circle(result_image, (x, y), radius, 255, 1)

            # Keep track of best circle (x, y, radius)
            if combined_circularity > best_circularity:
                best_circularity = combined_circularity
                detected_circle = (x, y, radius)

    return result_image, detected_circle


class CameraDeviationCalibrator:
    """
    Camera deviation calibrator that captures frames from Hikvision camera,
    detects circles, and records offset data.
    """

    def __init__(self, capture_id: int = 0,
                 data_shape: Tuple[int, int, int] = (360, 270, 3),
                 fps: float = 25.0,
                 pixel_format: str = "RGB8",
                 exposure_time: float = 5000.0,
                 gain: float = 16.9807,
                 gamma: float = 0.68):
        """
        Initialize the camera deviation calibrator.

        Args:
            capture_id: Physical camera index
            data_shape: Shape of camera frames (height, width, channels)
            fps: Target frames per second
            pixel_format: Pixel format for Hikvision camera
            exposure_time: Exposure time in microseconds
            gain: Gain in dB
            gamma: Gamma value
        """
        self.capture_id = capture_id
        self.data_shape = data_shape
        self.fps = fps

        # Store detected circles (x, y, radius) in pixel coordinates
        self.detected_circles: List[Tuple[int, int, int]] = []

        # Frame counter
        self.frame_count = 0
        self.detection_count = 0

        # Flag to control the animation loop
        self.running = False

        # Initialize matplotlib figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.ax.set_title("Camera Deviation Calibration")
        self.im = None
        self.info_text = None

        # Initialize HIK camera
        logger.info(f"Initializing Hikvision capture index {capture_id}...")
        self.camera = HikCameraDevice(
            device_class_id=0,
            capture_id=capture_id,
            state_feedback={
                "fps": fps,
                "buffer_size": 1,
                "hardware_latency_ms": 80.0,
                "format": {
                    "shape": list(data_shape),
                    "dtype": "uint8",
                    "type": "rgb",
                },
            },
            pixel_format=pixel_format,
            frame_rate_enable=True,
            exposure_time=exposure_time,
            gain=gain,
            gamma_enable=True,
            gamma=gamma,
            rotation_angle=90,
            load_calibration=False,
        )
        logger.info("Camera initialized successfully")

    def _flush_camera_buffer(self) -> None:
        """
        Flush old frames from camera buffer by reading and discarding them.
        Uses short timeout to quickly drain the buffer.
        """
        from third_party.MvImport.MvCameraControl_class import MV_FRAME_OUT, MV_OK

        max_flush = 10  # Maximum frames to flush
        flushed = 0

        while flushed < max_flush:
            stFrameInfo = MV_FRAME_OUT()
            # Use very short timeout (1ms) to check if there are buffered frames
            ret = self.camera.cam.MV_CC_GetImageBuffer(stFrameInfo, 1)

            if ret != MV_OK:
                # No more frames in buffer
                break

            # Free the frame immediately without processing
            self.camera.cam.MV_CC_FreeImageBuffer(stFrameInfo)
            flushed += 1

        if flushed > 0:
            logger.debug(f"Flushed {flushed} old frames from camera buffer")

    def _capture_single_frame(self) -> Optional[np.ndarray]:
        """
        Capture the latest frame from the camera.
        First flushes old buffered frames to minimize latency.

        Returns:
            Captured frame as numpy array, or None if capture failed
        """
        # Flush old frames first to get the latest one
        self._flush_camera_buffer()

        # Now capture the latest frame
        timestamp_ns, frame = self.camera._capture_frame()
        if frame is not None:
            # Ensure frame is in RGB format for visualization
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            return frame
        return None

    def update_plot(self, frame: int) -> List:
        """
        Update the matplotlib plot with new frame data.

        Args:
            frame: Frame number (from FuncAnimation)

        Returns:
            List of artists to update
        """
        if not self.running:
            return []

        # Capture frame
        captured_frame = self._capture_single_frame()

        if captured_frame is None:
            return []

        self.frame_count += 1

        # Detect circles and draw on frame
        frame_with_circles, detected_circle = detect_and_draw_circles_static(
            captured_frame, draw_circles=True
        )

        # If circle detected, store (x, y, r) and add visual markers
        if detected_circle is not None:
            self.detected_circles.append(detected_circle)
            self.detection_count += 1

            # Draw crosshair at detected center
            x, y, r = detected_circle
            cv2.drawMarker(frame_with_circles, (x, y), (0, 255, 0),
                         cv2.MARKER_CROSS, 20, 2)

        # Calculate statistics if we have detections
        median_circle = None
        std_circle = None
        offset_from_center = None

        if len(self.detected_circles) > 0:
            circles_array = np.array(self.detected_circles)
            median_circle = np.median(circles_array, axis=0)
            std_circle = circles_array.std(axis=0)

            # Calculate offset from image center
            height, width = self.data_shape[0], self.data_shape[1]
            image_center = np.array([width / 2.0, height / 2.0])
            offset_from_center = median_circle[:2] - image_center

        # Update image
        if self.im is None:
            # First time: create image plot
            self.ax.clear()
            self.im = self.ax.imshow(frame_with_circles, aspect='equal')
            self.ax.set_title("Camera Deviation Calibration")
            self.ax.axis('off')
        else:
            # Update existing image
            self.im.set_array(frame_with_circles)

        # Build info text
        info_text = f"Frames: {self.frame_count}\n"
        info_text += f"Detections: {self.detection_count}\n"

        if detected_circle is not None:
            x, y, r = detected_circle
            info_text += f"\nCurrent (x,y,r):\n  ({x}, {y}, {r})\n"

        if median_circle is not None and std_circle is not None:
            info_text += f"\nMedian (x,y,r):\n  ({median_circle[0]:.1f} ± {std_circle[0]:.1f},\n"
            info_text += f"   {median_circle[1]:.1f} ± {std_circle[1]:.1f},\n"
            info_text += f"   {median_circle[2]:.1f} ± {std_circle[2]:.1f})\n"

        if offset_from_center is not None:
            info_text += f"\nOffset from Center:\n  ({offset_from_center[0]:.1f},\n   {offset_from_center[1]:.1f}) px\n"

        info_text += f"\nPress Ctrl+C or\nclose window to stop"

        # Update existing text or create new one (more efficient than remove+create)
        if self.info_text is None:
            # First time: create text object
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
        else:
            # Update existing text object (more efficient)
            self.info_text.set_text(info_text)

        return []

    def run_calibration(self) -> None:
        """
        Run the calibration process using matplotlib animation.
        """
        logger.info("Starting calibration process...")
        logger.info("Close the plot window or press Ctrl+C to stop and save offsets")

        # Start camera grabbing
        ret = self.camera.cam.MV_CC_StartGrabbing()
        if ret != 0:
            logger.error(f"Failed to start grabbing, error code: 0x{ret:x}")
            return

        self.running = True

        try:
            # Set up animation (update every 50ms = ~20 FPS display)
            ani = animation.FuncAnimation(
                self.fig,
                self.update_plot,
                interval=50,
                blit=False,
                cache_frame_data=False
            )

            # Show plot
            plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
            plt.show()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.running = False
            # Stop grabbing
            self.camera.cam.MV_CC_StopGrabbing()
            logger.info(f"Calibration stopped. Total frames: {self.frame_count}, Total detections: {self.detection_count}")

    def save_offsets(self, output_dir: str = "calibration/hikcamera/mask_offset") -> None:
        """
        Save detected circle parameters to JSON file.

        The output file is named camera_{serial_number}.json and contains
        a dictionary with the median x, y, r values of detected circles.

        Args:
            output_dir: Output directory path
        """
        if not self.detected_circles:
            logger.warning("No circles detected, nothing to save")
            return

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get camera serial number
        camera_serial = getattr(self.camera, 'camera_serial', 'unknown')

        # Calculate median values for x, y, r
        circles_array = np.array(self.detected_circles)
        median_x = float(np.median(circles_array[:, 0]))
        median_y = float(np.median(circles_array[:, 1]))
        median_r = float(np.median(circles_array[:, 2]))

        # Calculate standard deviations for reference
        std_x = float(np.std(circles_array[:, 0]))
        std_y = float(np.std(circles_array[:, 1]))
        std_r = float(np.std(circles_array[:, 2]))

        # Calculate image center (reference point)
        height, width = self.data_shape[0], self.data_shape[1]
        image_center_x = width / 2.0
        image_center_y = height / 2.0

        # Calculate offset from image center
        offset_x = median_x - image_center_x
        offset_y = median_y - image_center_y

        # Build output dictionary
        output_data: Dict[str, Any] = {
            "x": median_x,
            "y": median_y,
            "r": median_r,
            "metadata": {
                "camera_serial": camera_serial,
                "capture_id": self.capture_id,
                "image_shape": list(self.data_shape),
                "total_detections": len(self.detected_circles),
                "std_x": std_x,
                "std_y": std_y,
                "std_r": std_r,
                "image_center_x": image_center_x,
                "image_center_y": image_center_y,
                "offset_x": offset_x,
                "offset_y": offset_y,
                "calibration_time": datetime.now().isoformat()
            }
        }

        # Generate output file path
        output_file = os.path.join(output_dir, f"camera_{camera_serial}.json")

        # Write JSON file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Calibration saved to {output_file}")
        logger.info(f"Camera serial: {camera_serial}")
        logger.info(f"Median circle (x, y, r): ({median_x:.2f}, {median_y:.2f}, {median_r:.2f})")
        logger.info(f"Offset from image center: ({offset_x:.2f}, {offset_y:.2f}) px")

    def close(self) -> None:
        """Close camera and release resources."""
        self.camera.close()
        logger.info("Camera closed")


def main() -> None:
    """Main function to run camera deviation calibration."""
    parser = argparse.ArgumentParser(
        description="Camera Deviation Calibration - Detect circles and save offsets"
    )
    parser.add_argument("--capture-id", "-d", type=int, default=1,
                       help="Camera device ID (default: 0)")
    parser.add_argument("--width", "-w", type=int, default=270,
                       help="Image width (default: 270)")
    parser.add_argument("--height", "-H", type=int, default=360,
                       help="Image height (default: 360)")
    parser.add_argument("--fps", "-f", type=float, default=25.0,
                       help="Frames per second (default: 25.0)")
    parser.add_argument("--exposure", "-e", type=float, default=5000.0,
                       help="Exposure time in microseconds (default: 5000.0)")
    parser.add_argument("--gain", "-g", type=float, default=16.9807,
                       help="Gain in dB (default: 16.9807)")
    parser.add_argument("--gamma", "-G", type=float, default=0.68,
                       help="Gamma value (default: 0.68)")
    parser.add_argument("--output-dir", "-o", type=str,
                       default="calibration/hikcamera/offset",
                       help="Output directory path (default: calibration/hikcamera/offset)")

    args = parser.parse_args()

    logger.info("Camera Deviation Calibration")
    logger.info("=" * 50)
    logger.info(f"Device ID: {args.capture_id}")
    logger.info(f"Resolution: {args.width}x{args.height}")
    logger.info(f"FPS: {args.fps}")
    logger.info(f"Exposure: {args.exposure} μs")
    logger.info(f"Gain: {args.gain} dB")
    logger.info(f"Gamma: {args.gamma}")
    logger.info("")

    # Create calibrator
    data_shape = (args.height, args.width, 3)
    calibrator = CameraDeviationCalibrator(
        capture_id=args.capture_id,
        data_shape=data_shape,
        fps=args.fps,
        pixel_format="RGB8",
        exposure_time=args.exposure,
        gain=args.gain,
        gamma=args.gamma
    )

    try:
        # Run calibration
        calibrator.run_calibration()

        # Save offsets to JSON file named camera_{serial}.json
        calibrator.save_offsets(output_dir=args.output_dir)

    except Exception as e:
        logger.error(f"Error during calibration: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        calibrator.close()


if __name__ == "__main__":
    main()
