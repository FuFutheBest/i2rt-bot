#!/usr/bin/env python3
"""
Vive Tracker Device - Device for reading Vive Tracker 6D pose data using OpenXR.

This module provides a unified device that can read 6D pose data from one or multiple
Vive Trackers using a single OpenXR session and writes data to shared memory.

Based on the recommended OpenXR pattern: one process manages all trackers.

Refer to this url for tips on setup vive tracker on Ubuntu:
https://gist.github.com/DanielArnett/c9a56c9c7cc0def20648480bca1f6772

change for windows compatibility:

Author: Han Xue, Jun Lv, Assistant
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import time
import ctypes
from ctypes import cast, byref
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Mapping
from devices.config import StateFeedbackConfig
from utils.logger_config import logger
try:
    import xr
    from xr.utils.gl import ContextObject
    from xr.utils.gl.glfw_util import GLFWOffscreenContextProvider
    import xr.ext.HTCX.vive_tracker_interaction as vive_tracker_interaction
    import xr.ext.KHR.opengl_enable as opengl_enable
except ImportError:
    logger.warning("OpenXR not found. Vive Tracker device will not be available.")
    xr = None
    ContextObject = None
    GLFWOffscreenContextProvider = None
    vive_tracker_interaction = None
    opengl_enable = None

from devices.base import BaseDevice
from utils.shm_utils import get_dtype
from utils.time_control import precise_loop_timing

class ViveTrackerDevice(BaseDevice):
    """
    Unified Vive Tracker device that reads 6D pose data from one or multiple trackers
    and writes all data into a single shared memory.

    The data_shape will be (N, 7), where N is the number of trackers.
    """

    def __init__(self, device_class_id: int = 0, *,
                 state_feedback: StateFeedbackConfig | Mapping[str, Any],
                 tracker_roles: Optional[List[str]] = None,
                 ) -> None:
        """
        Initialize the Vive Tracker device using OpenXR.

        Args:
            device_class_id: Unique identifier for this device instance.
            state_feedback: Pose stream contract.
            tracker_roles: A list of roles for the trackers to manage (e.g., ['left_foot', 'right_foot']).
                         If None or empty, it will operate in single-tracker mode, returning the first available tracker.
        """
        self.device_name = "ViveTrackerDevice"

        # Determine operation mode and validate data_shape
        self.tracker_roles = tracker_roles if tracker_roles else []

        if self.tracker_roles:
            # Multi-tracker mode
            num_trackers = len(self.tracker_roles)
            expected_shape = (num_trackers, 7)
            state_shape = tuple(state_feedback.format.shape) if isinstance(state_feedback, StateFeedbackConfig) else None
            if state_shape is not None:
                if state_shape != expected_shape:
                    raise ValueError(
                        f"state_feedback.format.shape {state_shape} does not match the number of tracker_roles. "
                        f"Expected {expected_shape} for {num_trackers} trackers."
                    )

            self.role_to_index_map = {role: i for i, role in enumerate(self.tracker_roles)}
            logger.info(f"Initializing in multi-tracker mode for roles: {self.tracker_roles}")
        else:
            # Single-tracker mode
            expected_shape = (1, 7)
            state_shape = tuple(state_feedback.format.shape) if isinstance(state_feedback, StateFeedbackConfig) else None
            if state_shape is not None:
                if state_shape != expected_shape:
                    raise ValueError(
                        f"state_feedback.format.shape {state_shape} is invalid for single-tracker mode. "
                        f"Expected {expected_shape}."
                    )
            logger.info("Initializing in single-tracker mode, will use first available tracker.")

        super().__init__(
            device_class_id=device_class_id,
            device_name=self.device_name,
            state_feedback=state_feedback,
        )
        expected_shape = (
            (len(self.tracker_roles), 7) if self.tracker_roles else (1, 7)
        )
        if tuple(self.data_shape) != expected_shape:
            raise ValueError(
                f"state_feedback.format.shape {self.data_shape} does not match "
                f"tracker_roles; expected {expected_shape}"
            )

        # OpenXR related attributes
        self.context: Optional[ContextObject] = None
        self.instance: Optional[Any] = None
        self.session: Optional[Any] = None
        self.action_set: Optional[Any] = None
        self.tracker_pose_action: Optional[Any] = None
        self.reference_space: Optional[Any] = None

        # Tracker management
        self.all_possible_role_paths: List[str] = []
        self.tracker_action_spaces: Dict[str, Any] = {}

        # Initialize OpenXR
        self._init_openxr()

    def _init_openxr(self) -> None:
        """Initialize OpenXR using the unified pattern for all trackers."""
        try:
            # Initialize OpenXR context
            self.context = ContextObject(
                context_provider=GLFWOffscreenContextProvider(),
                instance_create_info=xr.InstanceCreateInfo(
                    enabled_extension_names=[
                        opengl_enable.EXTENSION_NAME,
                        vive_tracker_interaction.EXTENSION_NAME,
                    ],
                ),
            )

            self.context.__enter__()  # Enter context manager
            self.instance = self.context.instance
            self.session = self.context.session

            # Setup tracker actions and spaces
            self._setup_unified_tracker_system()

            logger.info(f"OpenXR initialized successfully for {self.device_name}")

        except Exception as e:
            logger.error(f"Error initializing OpenXR: {e}")
            logger.warning("Device will run with zero pose data.")
            self.context = None

    def _setup_unified_tracker_system(self) -> None:
        """Setup the unified tracker system following the recommended OpenXR pattern."""
        try:
            # Define all possible Vive Tracker role paths
            self.all_possible_role_paths = [
                "/user/vive_tracker_htcx/role/waist",
                "/user/vive_tracker_htcx/role/left_foot",
                "/user/vive_tracker_htcx/role/right_foot",
                "/user/vive_tracker_htcx/role/left_shoulder",
                "/user/vive_tracker_htcx/role/right_shoulder",
                "/user/vive_tracker_htcx/role/chest",
                "/user/vive_tracker_htcx/role/camera",
                "/user/vive_tracker_htcx/role/left_elbow",
                "/user/vive_tracker_htcx/role/right_elbow",
                "/user/vive_tracker_htcx/role/left_knee",
                "/user/vive_tracker_htcx/role/right_knee",
                "/user/vive_tracker_htcx/role/handheld_object",
                "/user/vive_tracker_htcx/role/keyboard",
            ]

            # Convert string paths to OpenXR path handles
            tracker_subaction_paths = [
                xr.string_to_path(self.instance, path) for path in self.all_possible_role_paths
            ]

            # Use default action set from ContextObject so it gets attached automatically
            self.action_set = self.context.default_action_set

            # Create a single PoseAction for all trackers
            self.tracker_pose_action = xr.create_action(
                action_set=self.action_set,
                create_info=xr.ActionCreateInfo(
                    action_name="tracker_poses",
                    localized_action_name="Tracker Poses",
                    action_type=xr.ActionType.POSE_INPUT,
                    count_subaction_paths=len(tracker_subaction_paths),
                    subaction_paths=(xr.Path * len(tracker_subaction_paths))(*tracker_subaction_paths),
                ),
            )

            # Suggest bindings
            vive_tracker_profile_path = xr.string_to_path(
                self.instance, "/interaction_profiles/htc/vive_tracker_htcx"
            )

            bindings = [
                xr.ActionSuggestedBinding(
                    action=self.tracker_pose_action,
                    binding=xr.string_to_path(self.instance, f"{role_path}/input/grip/pose"),
                )
                for role_path in self.all_possible_role_paths
            ]

            xr.suggest_interaction_profile_bindings(
                instance=self.instance,
                suggested_bindings=xr.InteractionProfileSuggestedBinding(
                    interaction_profile=vive_tracker_profile_path,
                    count_suggested_bindings=len(bindings),
                    suggested_bindings=(xr.ActionSuggestedBinding * len(bindings))(*bindings),
                ),
            )

            # Create ActionSpaces for each tracker role
            for role_path_str, subaction_path_handle in zip(self.all_possible_role_paths, tracker_subaction_paths):
                try:
                    space = xr.create_action_space(
                        session=self.session,
                        create_info=xr.ActionSpaceCreateInfo(
                            action=self.tracker_pose_action,
                            subaction_path=subaction_path_handle,
                        ),
                    )
                    self.tracker_action_spaces[role_path_str] = space
                except Exception as e:
                    logger.debug(f"Could not create action space for {role_path_str}: {e}")

            # Create reference space
            self.reference_space = xr.create_reference_space(
                session=self.session,
                create_info=xr.ReferenceSpaceCreateInfo(
                    reference_space_type=xr.ReferenceSpaceType.LOCAL,
                    pose_in_reference_space=xr.Posef()
                ),
            )

            logger.info(f"Setup {len(self.tracker_action_spaces)} tracker action spaces")

        except Exception as e:
            logger.error(f"Error setting up unified tracker system: {e}")
            self.tracker_action_spaces = {}

    def _get_all_tracker_poses(self, frame_state) -> Dict[str, np.ndarray]:
        """
        Get poses for all available trackers using the unified OpenXR pattern.

        Args:
            frame_state: OpenXR frame state with predicted display time

        Returns:
            Dict mapping role path strings to pose arrays [x, y, z, qx, qy, qz, qw]
        """
        if self.context is None or not self.tracker_action_spaces:
            return {}

        tracker_poses = {}

        try:
            # Use the predicted display time from frame state
            predicted_time = frame_state.predicted_display_time

            # Query all tracker action spaces
            for role_path_str, space in self.tracker_action_spaces.items():
                try:
                    space_location = xr.locate_space(
                        space=space,
                        base_space=self.context.space,
                        time=predicted_time,
                    )

                    # Check if both position and orientation are valid
                    position_valid = bool(space_location.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT)
                    logger.debug(f"Position valid: {position_valid}")
                    orientation_valid = bool(space_location.location_flags & xr.SPACE_LOCATION_ORIENTATION_VALID_BIT)
                    logger.debug(f"Orientation valid: {orientation_valid}")

                    if position_valid and orientation_valid:
                        # Extract position and quaternion
                        p = space_location.pose.position
                        o = space_location.pose.orientation

                        # Create pose array in [x, y, z, qx, qy, qz, qw] format
                        numpy_data_dtype = get_dtype(self.data_dtype)
                        pose_data = np.array([
                            float(p.x), float(p.y), float(p.z),
                            float(o.x), float(o.y), float(o.z), float(o.w)
                        ], dtype=numpy_data_dtype)

                        tracker_poses[role_path_str] = pose_data

                except Exception as e:
                    logger.debug(f"Error reading tracker {role_path_str}: {e}")
                    continue

        except Exception as e:
            logger.debug(f"Error getting tracker poses: {e}")

        return tracker_poses

    def start_server(self) -> None:
        """Start the unified Vive Tracker device server."""
        # The base class start_server is almost perfect, but it calls _generate_random_array,
        # which we don't want. We want to read from OpenXR. So we override it.
        if self.running:
            logger.warning(f"Device {self.device_name}_{self.device_class_id} is already running")
            return

        logger.info(f"Starting {self.device_name}_{self.device_class_id} server...")
        self.running = True

        # Create shared memory
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create primary shared memory")

        logger.info(f"Server started. Primary shared memory: {self.shared_memory_name}")
        logger.info(f"Buffer configuration: {self.buffer_size} frames, {self.frame_size:,} bytes per frame")

        if self.context is None:
            self.running = False
            raise RuntimeError("OpenXR context not initialized. Cannot start tracker device.")

        # Run the unified OpenXR frame loop
        self._run_unified_frame_loop()

    def _run_unified_frame_loop(self) -> None:
        if sys.platform == 'win32':
            logger.info("Detected Windows platform. Using Windows-specific logic (Hardware Blocking).")
            self._run_unified_frame_loop_windows()
        else:
            # linux or linux2
            logger.info(f"Detected {sys.platform} platform. Using Linux logic (Precise Timing).")
            self._run_unified_frame_loop_linux()


    def _run_unified_frame_loop_windows(self) -> None:
        """Run the unified OpenXR frame loop and write data to a single SHM."""
        try:
            frame_count = 0
            session_was_focused = False
            logger.info("Starting unified OpenXR frame loop...")

            # Pre-allocate array for performance
            numpy_data_dtype = get_dtype(self.data_dtype)
            output_array = np.zeros(self.data_shape, dtype=numpy_data_dtype)

            # frame_loop is a generator. It handles xrWaitFrame internally.
            # If context is None, it won't even start.
            for frame_index, frame_state in enumerate(self.context.frame_loop() if self.context else []):
                if not self.running:
                    print("not running")
                    break

                frame_count += 1

                try:
                    # Check if logic should be processed:
                    # Including SYNCHRONIZED (3) and VISIBLE (4) is to maintain heartbeat, letting Runtime know the app is still alive
                    should_keep_alive = self.context.session_state in [
                        xr.SessionState.FOCUSED,
                        xr.SessionState.VISIBLE,
                        xr.SessionState.SYNCHRONIZED
                    ]

                    if should_keep_alive:
                        # === Key fix: Only FOCUSED state has permission to read input ===
                        if self.context.session_state == xr.SessionState.FOCUSED:
                            session_was_focused = True

                            # 1. Sync actions (only called when Focused)
                            active_action_set = xr.ActiveActionSet(
                                action_set=self.action_set,
                                subaction_path=xr.NULL_PATH,
                            )
                            xr.sync_actions(
                                session=self.session,
                                sync_info=xr.ActionsSyncInfo(
                                    count_active_action_sets=1,
                                    active_action_sets=ctypes.pointer(active_action_set),
                                ),
                            )
                            # 2. Get all tracker poses
                            tracker_poses = self._get_all_tracker_poses(frame_state)
                            timestamp_ns = time.time_ns()
                            # Reset output array to zeros
                            output_array.fill(0)

                            if not self.tracker_roles: # Single-tracker mode
                                if tracker_poses:
                                    # Get the first available pose
                                    first_pose = next(iter(tracker_poses.values()))
                                    output_array[0, :] = first_pose

                            else: # Multi-tracker mode
                                for role_path, pose_data in tracker_poses.items():
                                    role_name = role_path.split('/')[-1]
                                    if role_name in self.role_to_index_map:
                                        idx = self.role_to_index_map[role_name]
                                        output_array[idx, :] = pose_data

                            # Check for all-zero data (only considered abnormal when FOCUSED)
                            if np.all(output_array == 0):
                                if frame_count % 100 == 0:
                                    logger.warning("All tracker poses are zero (Session is FOCUSED). Check hardware tracking status.")
                            self._write_array_to_shm_with_timestamp(output_array, timestamp_ns)

                        else:
                            # === State is SYNCHRONIZED (3) or VISIBLE (4) ===
                            # Must allow loop to continue (wait/begin/end frame), but cannot read input
                            if frame_count % 200 == 0:
                                logger.warning(f"no focused, session state: {self.context.session_state}")

                            # Write all-zero data to prevent downstream readers from blocking or erroring
                            timestamp_ns = time.time_ns()
                            output_array.fill(0)
                            self._write_array_to_shm_with_timestamp(output_array, timestamp_ns)

                    else:
                        # Not ready yet (IDLE, READY, STOPPING, etc.)
                        if frame_count % 100 == 0:
                            logger.warning(f"no focused, session state: {self.context.session_state}")

                        timestamp_ns = time.time_ns()
                        output_array.fill(0)
                        self._write_array_to_shm_with_timestamp(output_array, timestamp_ns)

                except Exception as e:
                    logger.error(f"Error in frame {frame_index}: {e}")
                    continue

            # This part runs if the loop finishes without break
            if not session_was_focused and self.context:
                logger.warning("OpenXR session never entered FOCUSED state. Make sure headset is connected and worn.")

            # If the loop never ran, this indicates a serious initialization problem
            if self.running and frame_count == 0:
                logger.error("OpenXR frame loop did not run. This indicates OpenXR initialization or runtime failure.")
                logger.error("Device cannot function without OpenXR frame loop. Please check OpenXR setup and hardware connection.")

        except Exception as e:
            logger.error(f"Fatal error in unified OpenXR frame loop: {e}")
        finally:
            logger.info(f"Unified OpenXR frame loop ended after {frame_count} frames")

    def _run_unified_frame_loop_linux(self) -> None:
        """Run the unified OpenXR frame loop and write data to a single SHM."""
        try:
            # Create precise timing function
            wait_for_next_iteration = precise_loop_timing(self.update_interval)

            frame_count = 0
            session_was_focused = False

            logger.info("Starting unified OpenXR frame loop...")

            # Pre-allocate array for performance
            numpy_data_dtype = get_dtype(self.data_dtype)
            output_array = np.zeros(self.data_shape, dtype=numpy_data_dtype)

            # frame_loop is a generator. If context is None, it won't even start.
            for frame_index, frame_state in enumerate(self.context.frame_loop() if self.context else []):
                if not self.running:
                    break

                frame_count += 1

                try:
                    if self.context.session_state == xr.SessionState.FOCUSED:
                        session_was_focused = True
                        # Sync actions for the active (default) action set before querying poses
                        active_action_set = xr.ActiveActionSet(
                            action_set=self.action_set,
                            subaction_path=xr.NULL_PATH,
                        )
                        xr.sync_actions(
                            session=self.session,
                            sync_info=xr.ActionsSyncInfo(
                                count_active_action_sets=1,
                                active_action_sets=ctypes.pointer(active_action_set),
                            ),
                        )

                        # Get all tracker poses
                        tracker_poses = self._get_all_tracker_poses(frame_state)
                        timestamp_ns = time.time_ns()

                        # Reset output array to zeros
                        output_array.fill(0)

                        if not self.tracker_roles: # Single-tracker mode
                            if tracker_poses:
                                # Get the first available pose
                                first_pose = next(iter(tracker_poses.values()))
                                output_array[0, :] = first_pose
                        else: # Multi-tracker mode
                            for role_path, pose_data in tracker_poses.items():
                                role_name = role_path.split('/')[-1]
                                if role_name in self.role_to_index_map:
                                    idx = self.role_to_index_map[role_name]
                                    output_array[idx, :] = pose_data

                        self._write_array_to_shm_with_timestamp(output_array, timestamp_ns)

                    else: # Not FOCUSED, write zeros
                        timestamp_ns = time.time_ns()
                        output_array.fill(0)
                        self._write_array_to_shm_with_timestamp(output_array, timestamp_ns)

                    # Wait for next iteration using precise timing
                    wait_for_next_iteration()

                    # Log progress occasionally
                    if frame_count % 900 == 0:  # Every ~10 seconds at 90 FPS
                        active_trackers = len(tracker_poses) if 'tracker_poses' in locals() else 0
                        logger.debug(f"Frame {frame_count}: {active_trackers} active trackers found")

                except Exception as e:
                    logger.error(f"Error in frame {frame_index}: {e}")
                    continue

            # This part runs if the loop finishes without break, or if context was None from the start
            if not session_was_focused and self.context:
                logger.warning("OpenXR session never entered FOCUSED state. Make sure headset is connected and worn.")

            # If the loop never ran (e.g., no context), we still need to provide zero data if running
            if self.running and frame_count == 0:
                logger.warning("OpenXR frame loop did not run. Writing zero data.")
                while self.running:
                    timestamp_ns = time.time_ns()
                    output_array.fill(0)
                    self._write_array_to_shm_with_timestamp(output_array, timestamp_ns)
                    wait_for_next_iteration()

        except Exception as e:
            logger.error(f"Error in unified OpenXR frame loop: {e}")
        finally:
            logger.info(f"Unified OpenXR frame loop ended after {frame_count} frames")


    def stop_server(self) -> None:
        """Stop the unified Vive Tracker device server."""
        if not self.running:
            return

        logger.info(f"Stopping {self.device_name}_{self.device_class_id} server...")
        self.running = False

        # Clean up OpenXR
        if self.context:
            try:
                # This is a generator, so we can't just call __exit__.
                # Setting self.running to False and letting the loop break is the way.
                # However, if the server is stopped from outside, we need to ensure cleanup.
                # The context is a context manager, so __exit__ should be called.
                self.context.__exit__(None, None, None)
                logger.info("OpenXR context cleanup complete")
            except Exception as e:
                logger.error(f"Error cleaning up OpenXR context: {e}")

        # Clean up primary shared memory (this is the only one now)
        self._cleanup_shared_memory()
        logger.info("Server stopped")

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        info = super().get_device_info()
        info.update({
            "tracker_roles": self.tracker_roles,
            "openxr_initialized": self.context is not None,
            "tracker_action_spaces": len(self.tracker_action_spaces),
            "quaternion_format": "(x, y, z, w)",
            "managed_trackers": len(self.tracker_roles) if self.tracker_roles else 1,
        })
        return info

    def __del__(self) -> None:
        """Destructor to ensure proper cleanup."""
        try:
            if hasattr(self, 'context') and self.context:
                self.context.__exit__(None, None, None)
        except:
            pass
        super().__del__()



def main() -> None:
    """Main function to run the unified Vive Tracker device server."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Unified Vive Tracker device server using OpenXR")
    parser.add_argument("--device-class-id", "-d", type=int, default=0,
                        help="Device ID (default: 0)")
    parser.add_argument("--fps", "-f", type=float, default=90.0,
                        help="Frames per second (default: 90.0)")
    parser.add_argument("--tracker-roles", "-r", type=str, default='left_foot,right_foot',
                        help="Comma-separated list of tracker roles (e.g., 'left_foot,right_foot')")
    parser.add_argument("--buffer-size", "-b", type=int, default=100,
                        help="Buffer size in frames (default: 100)")
    parser.add_argument("--hardware-latency", "-l", type=float, default=5.0,
                        help="Hardware latency in milliseconds (default: 5.0)")
    parser.add_argument("--data-dtype", type=str, default="float64",
                        help="Data type for pose data (default: float64)")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Optional device config file to load settings from (overrides command line args)",
    )

    args = parser.parse_args()

    # Load from config file if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_data = yaml.safe_load(f)

            # Find the ViveTrackerDevice config
            vive_config = None
            for device_config in config_data.get('devices', []):
                if device_config.get('class') == 'ViveTrackerDevice':
                    vive_config = device_config
                    break

            if vive_config:
                logger.info(f"Loading settings from {args.config}")
                device_class_id = vive_config.get('device_class_id', args.device_class_id)
                tracker_roles = vive_config.get('tracker_roles', args.tracker_roles)
                state_feedback = vive_config['state_feedback']
            else:
                raise ValueError("No ViveTrackerDevice found in config file")

        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            return
    else:
        device_class_id = args.device_class_id
        tracker_roles = args.tracker_roles.split(',') if args.tracker_roles else None
        state_feedback = {
            "fps": args.fps,
            "buffer_size": args.buffer_size,
            "hardware_latency_ms": args.hardware_latency,
            "format": {
                "shape": [len(tracker_roles), 7] if tracker_roles else [1, 7],
                "dtype": args.data_dtype,
                "type": ["xyz", "qxqyqzqw"] * (len(tracker_roles) if tracker_roles else 1),
            },
        }

    # Create device with parsed arguments
    device = ViveTrackerDevice(
        device_class_id=device_class_id,
        state_feedback=state_feedback,
        tracker_roles=tracker_roles,
    )

    logger.info("Unified Vive Tracker Device Server (OpenXR)")
    logger.info("==========================================")
    logger.info(f"Device ID: {device_class_id}")
    logger.info(f"FPS: {device.fps}")
    if device.tracker_roles:
        logger.info(f"Mode: Multi-tracker ({len(device.tracker_roles)} trackers)")
        logger.info(f"Tracker Roles: {device.tracker_roles}")
    else:
        logger.info(f"Mode: Single tracker (first available)")
    logger.info(f"Data Shape: {device.data_shape}")
    logger.info(f"Buffer size: {device.buffer_size} frames")
    logger.info(f"Hardware latency: {device.hardware_latency_ms} ms")
    logger.info(f"Data type: {device.data_dtype}")
    logger.info(f"Quaternion format: (x, y, z, w)")

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
