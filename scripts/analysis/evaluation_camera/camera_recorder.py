#!/usr/bin/env python3
"""
Camera Recorder Script
Camera viewer and recorder that directly reads camera ID through OpenCV and visualizes
Supports real-time recording with configurable resolution, FPS, and display scaling

Author: Assistant
"""

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

try:
    import pyrealsense2 as rs  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    rs = None  # type: ignore[assignment]


class CameraRecorder:
    """
    Camera recorder that directly uses OpenCV to read camera and record videos
    """

    def __init__(self, camera_id: int, save_path: str = "./recordings",
                 width: Optional[int] = None, height: Optional[int] = None,
                 fps: Optional[float] = None, scale: float = 1.0,
                 show_window: bool = True, enable_recording: bool = True,
                 auto_start_recording: bool = False, record_key: str = "r",
                 window_name: Optional[str] = None, video_format: str = "mp4",
                 fourcc: Optional[str] = None,
                 use_realsense: bool = False,
                 realsense_serial: Optional[str] = None):
        """
        Initialize camera viewer

        Args:
            camera_id: Camera device ID
            save_path: Path to save recorded videos
            width: Desired frame width (None to use camera default)
            height: Desired frame height (None to use camera default)
            fps: Desired frame rate (None to use camera default)
            scale: Scale factor for display window (1.0 = original size)
            show_window: Whether to display the OpenCV window
            enable_recording: Whether recording functionality is enabled
            auto_start_recording: Whether to start recording automatically
            record_key: Keyboard key to toggle recording when window is shown
            window_name: Custom window name for display
            use_realsense: Whether to acquire frames via Intel RealSense SDK
            realsense_serial: Optional RealSense serial number to lock to a device
            video_format: File extension for recorded video (e.g., mp4, avi)
            fourcc: Optional override for codec used by OpenCV's VideoWriter
        """
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        self.use_realsense = use_realsense
        self.realsense_serial = realsense_serial

        if self.use_realsense and rs is None:
            raise RuntimeError(
                "pyrealsense2 is not installed; install it or disable the RealSense mode."
            )

        # Camera settings
        self.desired_width = width
        self.desired_height = height
        self.desired_fps = fps
        self.scale = scale
        self.show_window = show_window
        self.enable_recording = enable_recording
        self.auto_start_recording = auto_start_recording
        if not record_key:
            record_key = "r"
        self.record_key = record_key.lower()
        self.window_name = window_name if window_name else f"Camera {self.camera_id}"
        self.current_width: Optional[int] = None
        self.current_height: Optional[int] = None
        self.current_fps: Optional[float] = None
        self.video_format = (video_format or "mp4").lower()
        default_fourcc_map = {
            "avi": "XVID",
            "mp4": "mp4v",
            "mov": "avc1",
            "mkv": "X264"
        }
        selected_fourcc = fourcc if fourcc else default_fourcc_map.get(self.video_format, "mp4v")
        self.fourcc_code = selected_fourcc

        # Statistics
        self.frame_count = 0
        self.start_time = None

        # Recording related
        self.is_recording = False
        self.video_writer = None
        self.save_path = Path(save_path)
        self.current_recording_path = None
        self.recording_frame_count = 0
        self.recording_start_time = None

        # RealSense specific state
        self.rs_pipeline = None
        self.rs_config = None
        self.rs_profile = None

        # Create save directory if it doesn't exist
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def connect(self) -> bool:
        """
        Connect to camera

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            print(f"Connecting to camera device {self.camera_id}...")
            if self.enable_recording:
                print(f"Recording enabled. Save path: {self.save_path}")
            else:
                print("Recording disabled. Running in display-only mode")

            if self.show_window:
                print(f"Display window: {self.window_name} (scale: {self.scale})")
            else:
                print("Display disabled. Running in headless mode")

            if self.auto_start_recording and self.enable_recording:
                print("Recording will auto-start after connection")
            print(self.use_realsense)
            connected = self._connect_realsense() if self.use_realsense else self._connect_opencv()
            if not connected:
                return False

            width = self.current_width or self.desired_width or 0
            height = self.current_height or self.desired_height or 0
            fps = self.current_fps if self.current_fps not in (None, 0.0) else (self.desired_fps or 0.0)

            source_name = (
                f"RealSense camera {self.realsense_serial or self.camera_id}"
                if self.use_realsense else
                f"camera device {self.camera_id}"
            )

            print(f"Successfully connected to {source_name}")
            print(f"Resolution: {width}x{height}")
            print(f"FPS: {fps:.1f}")

            # Warn if requested settings differ from actual
            if self.desired_width is not None and width != self.desired_width:
                print(f"Warning: Requested width {self.desired_width}, but got {width}")
            if self.desired_height is not None and height != self.desired_height:
                print(f"Warning: Requested height {self.desired_height}, but got {height}")
            if self.desired_fps is not None and self.current_fps is not None:
                if abs(self.current_fps - self.desired_fps) > 0.1:
                    print(f"Warning: Requested FPS {self.desired_fps:.1f}, but got {self.current_fps:.1f}")

            return True

        except Exception as e:
            print(f"Error occurred while connecting to camera: {e}")
            return False

    def start_recording(self) -> bool:
        """
        Start recording video

        Returns:
            bool: True if recording started successfully, False otherwise
        """
        if not self.enable_recording:
            print("Recording is disabled")
            return False

        if self.is_recording:
            print("Already recording!")
            return False

        if not self._is_camera_ready():
            print("Camera not connected!")
            return False

        try:
            # Get video properties
            width = int(self.current_width or self.desired_width or 640)
            height = int(self.current_height or self.desired_height or 480)
            camera_fps = self.current_fps if self.current_fps not in (None, 0.0) else 0.0

            # Use desired FPS for recording, fallback to camera FPS or default
            if self.desired_fps is not None:
                fps = self.desired_fps
            elif camera_fps > 0:
                fps = camera_fps
            else:
                fps = 30.0  # Default to 30 FPS if cannot get FPS

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_{self.camera_id}_{timestamp}.mp4"
            self.current_recording_path = self.save_path / filename

            # Create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(self.current_recording_path),
                fourcc,
                fps,
                (width, height)
            )

            if not self.video_writer.isOpened():
                print("Error: Cannot create video writer!")
                self.video_writer = None
                return False

            self.is_recording = True
            self.recording_frame_count = 0
            self.recording_start_time = time.time()

            print(f"Started recording to: {self.current_recording_path}")
            print(f"Recording parameters: {width}x{height} @ {fps:.1f} FPS")
            return True

        except Exception as e:
            print(f"Error occurred while starting recording: {e}")
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            return False

    def stop_recording(self) -> None:
        """Stop recording video"""
        if not self.is_recording:
            print("Not recording!")
            return

        self.is_recording = False

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        if self.recording_start_time:
            duration = time.time() - self.recording_start_time
            print(f"Recording stopped")
            print(f"Saved {self.recording_frame_count} frames ({duration:.2f} seconds)")
            print(f"Video saved to: {self.current_recording_path}")

        self.recording_frame_count = 0
        self.recording_start_time = None

    def start_display(self, ready_conn: Optional[Any] = None) -> None:
        """
        Start displaying camera images
        """
        if not self.connect():
            if ready_conn is not None:
                try:
                    ready_conn.send({'status': 'failed', 'error': 'connect_failed'})
                except Exception:
                    pass
                finally:
                    ready_conn.close()
            return

        self.running = True
        self.start_time = time.time()

        print("Starting camera display")
        if self.show_window:
            print("Controls:")
            print("  'q' - Quit")
            print("  'c' - Toggle color/grayscale")
            if self.enable_recording:
                print(f"  '{self.record_key}' - Start/Stop recording")
        else:
            print("Running in headless mode (no display window)")
            if self.enable_recording:
                print("Recording can be controlled programmatically only")

        if self.auto_start_recording and self.enable_recording and not self.is_recording:
            self.start_recording()

        ready_conn_pending = ready_conn
        ready_conn = None  # ensure we don't reuse closed reference

        # Display mode: True for color, False for grayscale
        color_mode = True

        try:
            while self.running:
                # Read frame
                ret, frame = self._read_frame()

                if not ret:
                    print("Warning: Cannot read camera frame")
                    time.sleep(0.01)
                    continue

                self.frame_count += 1

                if ready_conn_pending is not None:
                    try:
                        ready_conn_pending.send({
                            'status': 'ready',
                            'record_key': self.record_key,
                            'auto_recording': self.is_recording,
                            'show_window': self.show_window
                        })
                    except Exception:
                        pass
                    finally:
                        try:
                            ready_conn_pending.close()
                        except Exception:
                            pass
                        ready_conn_pending = None
                    # Continue processing frame as normal after signaling readiness

                # Write frame to video if recording
                if self.is_recording and self.video_writer:
                    self.video_writer.write(frame)
                    self.recording_frame_count += 1

                # Prepare frame for display if needed
                key = -1
                if self.show_window:
                    display_frame = frame.copy()
                    if not color_mode:
                        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

                    if self.is_recording:
                        # Add recording indicator to display
                        recording_text = f"REC ({self.recording_frame_count} frames)"
                        cv2.putText(display_frame, recording_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        # Add red circle indicator
                        cv2.circle(display_frame, (display_frame.shape[1] - 30, 30), 15, (0, 0, 255), -1)

                    # Scale frame for display if needed
                    if self.scale != 1.0:
                        display_width = int(display_frame.shape[1] * self.scale)
                        display_height = int(display_frame.shape[0] * self.scale)
                        display_frame = cv2.resize(display_frame, (display_width, display_height))

                    # Display frame
                    cv2.imshow(self.window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    time.sleep(0.001)

                # Handle key presses
                if key == ord('q'):
                    print("User requested exit")
                    break
                elif key == ord('c'):
                    # Toggle color/grayscale mode
                    color_mode = not color_mode
                    print(f"Switched to {'color' if color_mode else 'grayscale'} mode")
                elif self.enable_recording and key == ord(self.record_key):
                    # Toggle recording
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording()

        except KeyboardInterrupt:
            print("Received interrupt signal")
        finally:
            self.stop()
            if ready_conn_pending is not None:
                try:
                    ready_conn_pending.send({'status': 'stopped'})
                except Exception:
                    pass
                finally:
                    try:
                        ready_conn_pending.close()
                    except Exception:
                        pass

    def stop(self) -> None:
        """Stop viewer"""
        self.running = False

        # Stop recording if still recording
        if self.is_recording:
            self.stop_recording()

        self._release_camera()
        if self.show_window:
            cv2.destroyAllWindows()
        print("Camera viewer stopped")

    def _signal_handler(self, signum, frame):
        """Signal handler"""
        print(f"Received signal {signum}, stopping...")
        self.running = False
        if self.is_recording:
            self.stop_recording()
        self._release_camera()
        if self.show_window:
            cv2.destroyAllWindows()
        print("Camera viewer stopped")
        sys.exit(0)

    # Internal helpers ---------------------------------------------------

    def _connect_opencv(self) -> bool:
        """Connect to a standard video device via OpenCV."""
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_id}")
            return False

        if self.desired_width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width)
        if self.desired_height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height)
        if self.desired_fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, self.desired_fps)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 0 else None

        self.current_width = width
        self.current_height = height
        self.current_fps = fps
        return True

    def _connect_realsense(self) -> bool:
        """Connect to an Intel RealSense device."""
        if rs is None:
            print("Error: pyrealsense2 is not installed.")
            return False

        requested = (
            int(self.desired_width) if self.desired_width else 1280,
            int(self.desired_height) if self.desired_height else 720,
            float(self.desired_fps) if self.desired_fps else 30.0
        )
        fallback_presets = [
            requested,
            (1280, 720, 30.0),
            (640, 480, 30.0)
        ]

        # Remove duplicates while preserving order
        seen: set[Tuple[int, int, float]] = set()
        attempts: list[Tuple[int, int, float]] = []
        for width, height, fps in fallback_presets:
            key = (int(width), int(height), float(fps))
            if key not in seen:
                attempts.append(key)
                seen.add(key)

        last_error = None
        for width, height, fps in attempts:
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                if self.realsense_serial:
                    config.enable_device(self.realsense_serial)
                config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, max(1, int(round(fps))))
                profile = pipeline.start(config)

                stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
                self.current_width = stream_profile.width()
                self.current_height = stream_profile.height()
                self.current_fps = float(stream_profile.fps())

                self.rs_pipeline = pipeline
                self.rs_config = config
                self.rs_profile = profile
                return True
            except Exception as exc:
                last_error = exc
                try:
                    pipeline.stop()  # type: ignore[attr-defined]
                except Exception:
                    pass
                self.rs_pipeline = None
                self.rs_config = None
                self.rs_profile = None

        print(f"Error: Cannot start RealSense camera {self.realsense_serial or self.camera_id}: {last_error}")
        return False

    def _read_realsense_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.rs_pipeline:
            return False, None
        try:
            frames = self.rs_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return False, None
            frame = np.asanyarray(color_frame.get_data())
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        except Exception as exc:
            print(f"Warning: Failed to read RealSense frame: {exc}")
            return False, None

    def _read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.use_realsense:
            return self._read_realsense_frame()
        if not self.cap:
            return False, None
        return self.cap.read()

    def _release_camera(self) -> None:
        if self.use_realsense:
            if self.rs_pipeline:
                try:
                    self.rs_pipeline.stop()
                except Exception:
                    pass
            self.rs_pipeline = None
            self.rs_config = None
            self.rs_profile = None
        else:
            if self.cap:
                self.cap.release()
            self.cap = None

    def _is_camera_ready(self) -> bool:
        if self.use_realsense:
            return self.rs_pipeline is not None
        return bool(self.cap and self.cap.isOpened())


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Camera recorder with real-time visualization and recording capability")
    parser.add_argument("--camera_id", type=int, required=True, help="Camera device ID")
    parser.add_argument("--save_path", type=str, default="./recordings",
                        help="Path to save recorded videos (default: ./recordings)")
    parser.add_argument("--width", type=int, default=1080,
                        help="Camera frame width (default: use camera default)")
    parser.add_argument("--height", type=int, default=640,
                        help="Camera frame height (default: use camera default)")
    parser.add_argument("--fps", type=float, default=30,
                        help="Camera frame rate (default: use camera default)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for display window (1.0 = original size)")
    parser.add_argument("--show", action="store_true",
                        help="Show live camera window (default: headless mode)")
    parser.add_argument("--use-realsense", type=bool, default=False)
    args = parser.parse_args()

    # Create recorder
    recorder = CameraRecorder(
        args.camera_id,
        args.save_path,
        width=args.width,
        height=args.height,
        fps=args.fps,
        scale=args.scale,
        show_window=args.show,
        use_realsense=args.use_realsense
    )

    # Start display
    recorder.start_display()


if __name__ == "__main__":
    main()
