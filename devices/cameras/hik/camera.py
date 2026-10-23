#!/usr/bin/env python3
"""Hikvision camera adapter implementation.

Author: Jun Lv, OpenAI Assistant
"""

import json
import os
import time
from ctypes import POINTER, byref, c_bool, c_ubyte, cast
from typing import Any, Dict, Mapping, Optional, Tuple

import cv2
import numpy as np
import psutil

from devices.cameras.base import BaseCameraDevice
from devices.config import StateFeedbackConfig
from devices.cameras.hik.sdk import (
    HIK_CAMERA_AVAILABLE,
    MV_ACCESS_Exclusive,
    MV_CC_DEVICE_INFO,
    MV_CC_DEVICE_INFO_LIST,
    MV_FRAME_OUT,
    MV_GENTL_CAMERALINK_DEVICE,
    MV_GENTL_CXP_DEVICE,
    MV_GENTL_GIGE_DEVICE,
    MV_GENTL_XOF_DEVICE,
    MV_GIGE_DEVICE,
    MV_OK,
    MV_TRIGGER_MODE_OFF,
    MV_USB_DEVICE,
    MVCC_ENUMVALUE,
    MVCC_FLOATVALUE,
    MVCC_INTVALUE,
    MVCC_STRINGVALUE,
    MvCamera,
    PixelType_Gvsp_BGR8_Packed,
    PixelType_Gvsp_BayerBG8,
    PixelType_Gvsp_BayerGB8,
    PixelType_Gvsp_BayerGR8,
    PixelType_Gvsp_BayerRG8,
    PixelType_Gvsp_Mono8,
    PixelType_Gvsp_RGB8_Packed,
)
from utils.logger_config import logger
from utils.time_control import precise_loop_timing

total_cores = psutil.cpu_count()
# Assign the last 8 cores to the server.
num_cores_to_bind = 8
cores_to_bind = set(range(total_cores - num_cores_to_bind, total_cores))
os.sched_setaffinity(0, cores_to_bind)


class HikCameraDevice(BaseCameraDevice):
    """
    Hikvision camera device class that captures video from Hikvision camera and writes to shared memory.
    Uses the MvCamera SDK for camera control and image acquisition.
    """

    def __init__(self, device_class_id: int = 0, *,
                 state_feedback: StateFeedbackConfig | Mapping[str, Any],
                 capture_id: int = 0,
                 pixel_format: Optional[str] = None,
                 frame_rate_enable: Optional[bool] = None,
                 exposure_time: Optional[float] = None,
                 gain: Optional[float] = None,
                 gamma_enable: Optional[bool] = None,
                 gamma: Optional[float] = None,
                 rotation_angle: int = 0,
                 load_calibration: bool = True):
        """
        Initialize the Hikvision camera device.

        Args:
            device_class_id: Logical ID for shared memory
            capture_id: Physical index in enumerated cameras
            data_shape: Shape of camera frames (height, width) for grayscale or (height, width, channels) for color
            fps: Target frames per second for capture
            data_dtype: Data type for camera frames (default: np.uint8)
            buffer_size: Number of frames to store in buffer (default: 1)
            hardware_latency_ms: Hardware latency in milliseconds (default: 0.0)
            pixel_format: Pixel format (e.g., "RGB8", "BGR8", "Mono8") (default: None)
            frame_rate_enable: Enable frame rate control (default: None)
            exposure_time: Exposure time in microseconds (default: None)
            gain: Gain in dB (default: None)
            gamma_enable: Enable gamma correction (default: None)
            gamma: Gamma value (default: None)
            rotation_angle: Image rotation angle in degrees (supports 0/90/180/270, clockwise)
            load_calibration: Whether to load circle calibration file for offset compensation (default: True)

        Raises:
            RuntimeError: If Hikvision camera modules are not available
            RuntimeError: If camera initialization fails
        """
        if not HIK_CAMERA_AVAILABLE:
            raise RuntimeError("Hikvision camera modules are not available. Please install MvImport.")

        self.capture_id = int(capture_id)
        super().__init__(device_class_id=device_class_id, state_feedback=state_feedback)

        # Override device name for Hikvision camera
        self.device_name = "HikCameraDevice"
        self.shared_memory_name = f"{self.device_name}_{self.device_class_id}_data"

        # Camera specific attributes
        self.cam = None
        self.is_color = len(self.data_shape) == 3 and self.data_shape[2] == 3
        self._resize_warned = False

        # Camera info (will be populated during initialization)
        self.camera_model = "Unknown"
        self.camera_serial = "Unknown"
        self.camera_manufacturer = "Unknown"

        # Circle calibration data (x, y, r) from calibration file
        self.circle_calibration: Optional[Dict[str, Any]] = None

        # Hikvision camera hardware parameters (None means not set)
        self.pixel_format = pixel_format
        self.frame_rate_enable = frame_rate_enable
        self.exposure_time = exposure_time
        self.gain = gain
        self.gamma_enable = gamma_enable
        self.gamma = gamma
        self.rotation_angle = self._validate_rotation(rotation_angle)
        self.load_calibration = load_calibration
        if self.rotation_angle:
            logger.info(f"Hikvision camera frames will be rotated {self.rotation_angle} degrees clockwise before sharing")
        if not self.load_calibration:
            logger.info("Hikvision camera calibration loading is disabled")

        # Initialize camera
        if not self._initialize_camera():
            raise RuntimeError(f"Failed to initialize Hikvision camera {device_class_id}")

    def _validate_rotation(self, rotation_angle: int) -> int:
        """Validate supported rotation angles."""
        valid_angles = {0, 90, 180, 270}
        if rotation_angle not in valid_angles:
            raise ValueError(f"Unsupported rotation_angle={rotation_angle}. Supported values: {sorted(valid_angles)}")
        return rotation_angle

    def _apply_rotation(self, image: np.ndarray) -> np.ndarray:
        """Rotate image according to configured rotation_angle."""
        if self.rotation_angle == 0:
            return image

        rotate_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }
        try:
            return cv2.rotate(image, rotate_map[self.rotation_angle])
        except Exception as exc:
            logger.error(f"Failed to rotate frame by {self.rotation_angle} degrees: {exc}")
            return image

    def _set_camera_parameters(self) -> bool:
        """
        Configure Hikvision camera parameters.

        Parameters are set in the following order (as required by Hikvision SDK):
        1. Pixel Format (RGB8) - Must be set first
        2. Acquisition Frame Rate Control Enable
        3. Acquisition Frame Rate
        4. Exposure Time
        5. Gain
        6. Gamma Enable
        7. Gamma

        Returns:
            bool: True if all parameters set successfully, False otherwise
        """
        if not self.cam:
            return False

        logger.info("=== Configuring Hikvision camera parameters ===")

        # 1. Set pixel format (must be set first!)
        if self.pixel_format is not None:
            logger.info(f"1. Setting pixel format to {self.pixel_format}...")
            ret = self.cam.MV_CC_SetEnumValueByString("PixelFormat", self.pixel_format)
            if ret != MV_OK:
                logger.warning(f"   Failed to set pixel format! ret[0x{ret:x}]")
                # Try with "Packed" suffix as fallback
                if "Packed" not in self.pixel_format:
                    logger.info(f"   Trying {self.pixel_format}Packed...")
                    ret = self.cam.MV_CC_SetEnumValueByString("PixelFormat", f"{self.pixel_format}Packed")
                    if ret != MV_OK:
                        logger.error(f"   Failed to set pixel format! ret[0x{ret:x}]")
                        return False
            logger.info("   ✓ Pixel format set successfully")

        # 2. Enable frame rate control
        if self.frame_rate_enable is not None:
            logger.info(f"2. {'Enabling' if self.frame_rate_enable else 'Disabling'} frame rate control...")
            ret = self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", self.frame_rate_enable)
            if ret != MV_OK:
                logger.warning(f"   Failed to set frame rate control! ret[0x{ret:x}]")
            else:
                logger.info("   ✓ Frame rate control configured successfully")

        # 3. Set frame rate
        if self.fps is not None:
            logger.info(f"3. Setting frame rate to {self.fps}...")
            ret = self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", self.fps)
            if ret != MV_OK:
                logger.warning(f"   Failed to set frame rate! ret[0x{ret:x}]")
            else:
                logger.info("   ✓ Frame rate set successfully")

        # 4. Set exposure time
        if self.exposure_time is not None:
            logger.info(f"4. Setting exposure time to {self.exposure_time} μs...")
            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", self.exposure_time)
            if ret != MV_OK:
                logger.warning(f"   Failed to set exposure time! ret[0x{ret:x}]")
            else:
                logger.info("   ✓ Exposure time set successfully")

        # 5. Set gain
        if self.gain is not None:
            logger.info(f"5. Setting gain to {self.gain} dB...")
            ret = self.cam.MV_CC_SetFloatValue("Gain", self.gain)
            if ret != MV_OK:
                logger.warning(f"   Failed to set gain! ret[0x{ret:x}]")
            else:
                logger.info("   ✓ Gain set successfully")

        # 6. Enable gamma correction
        if self.gamma_enable is not None:
            logger.info(f"6. {'Enabling' if self.gamma_enable else 'Disabling'} gamma correction...")
            # Try to set GammaSelector to User (some cameras require this)
            try:
                self.cam.MV_CC_SetEnumValue("GammaSelector", 1)  # 1 = User
            except Exception:
                pass

            ret = self.cam.MV_CC_SetBoolValue("GammaEnable", self.gamma_enable)
            if ret != MV_OK:
                logger.warning(f"   Failed to set gamma enable! ret[0x{ret:x}]")
            else:
                logger.info("   ✓ Gamma correction configured successfully")

        # 7. Set gamma value
        if self.gamma is not None:
            logger.info(f"7. Setting gamma value to {self.gamma}...")
            ret = self.cam.MV_CC_SetFloatValue("Gamma", self.gamma)
            if ret != MV_OK:
                logger.warning(f"   Failed to set gamma value! ret[0x{ret:x}]")
            else:
                logger.info("   ✓ Gamma value set successfully")

        logger.info("=== Camera parameters configuration completed ===")
        return True

    def _load_circle_calibration(self) -> None:
        """
        Load circle calibration data from JSON file.

        The calibration file should be located at:
        calibration/hikcamera/offset/camera_{serial_number}.json

        Raises:
            FileNotFoundError: If calibration file does not exist for this camera
        """
        calibration_dir = "calibration/hikcamera/offset"
        calibration_file = os.path.join(calibration_dir, f"camera_{self.camera_serial}.json")

        if not os.path.exists(calibration_file):
            raise FileNotFoundError(
                f"Circle calibration file not found for camera {self.camera_serial}. "
                f"Expected path: {calibration_file}. "
                f"Please run scripts/calibration/calib_cam_deviation.py with --capture-id {self.capture_id} to generate it."
            )

        try:
            with open(calibration_file, 'r') as f:
                self.circle_calibration = json.load(f)

            # Validate required keys
            required_keys = ['x', 'y', 'r']
            for key in required_keys:
                if key not in self.circle_calibration:
                    raise ValueError(f"Calibration file missing required key: {key}")

            logger.info(f"Loaded circle calibration for camera {self.camera_serial}: "
                       f"x={self.circle_calibration['x']:.2f}, "
                       f"y={self.circle_calibration['y']:.2f}, "
                       f"r={self.circle_calibration['r']:.2f}")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in calibration file {calibration_file}: {e}")

    def _apply_offset_compensation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply offset compensation to the image based on circle calibration data.

        The offset_x and offset_y represent the deviation of the detected circle center
        from the image center. To compensate, we translate the image in the opposite direction.

        Args:
            image: Input image to compensate

        Returns:
            Compensated image (or original image if calibration is not loaded)
        """
        if self.circle_calibration is None and not self.load_calibration:
            # Calibration not loaded - return original image without compensation
            return image

        # Get metadata
        metadata = self.circle_calibration.get('metadata', {})
        if not metadata:
            raise ValueError(
                f"Circle calibration file for camera {self.camera_serial} is missing 'metadata' field."
            )

        # Check if current image shape matches calibration image shape
        calibration_shape = metadata.get('image_shape', None)
        if calibration_shape is None:
            raise ValueError(
                f"Circle calibration file for camera {self.camera_serial} is missing 'image_shape' in metadata."
            )

        # Convert to tuple for comparison (height, width, channels)
        calibration_shape_tuple = tuple(calibration_shape)
        current_shape = image.shape

        # Check if shapes match - must be exactly the same
        if current_shape != calibration_shape_tuple:
            raise ValueError(
                f"Image shape mismatch for camera {self.camera_serial}: "
                f"current shape {current_shape} does not match calibration shape {calibration_shape_tuple}. "
                f"Please re-run calibration with the correct image shape."
            )

        # Extract offset values - must exist in calibration
        if 'offset_x' not in metadata or 'offset_y' not in metadata:
            raise ValueError(
                f"Circle calibration file for camera {self.camera_serial} is missing 'offset_x' or 'offset_y' in metadata."
            )
        offset_x = metadata['offset_x']
        offset_y = metadata['offset_y']

        # If no offset, return original image
        if abs(offset_x) < 0.5 and abs(offset_y) < 0.5:
            return image

        # Convert offset to integer for translation
        # offset_x < 0 means circle is left of center, so we translate image right (positive tx)
        # offset_y > 0 means circle is above center, so we translate image down (positive ty)
        tx = int(round(-offset_x))  # Negative because we want opposite direction
        ty = int(round(-offset_y))  # Negative because we want opposite direction

        if tx == 0 and ty == 0:
            return image

        # Create translation matrix
        # [1, 0, tx]
        # [0, 1, ty]
        M = np.float32([[1, 0, tx], [0, 1, ty]])

        # Apply translation using cv2.warpAffine
        # Use BORDER_REPLICATE to fill edges with replicated pixels
        height, width = image.shape[:2]
        compensated_image = cv2.warpAffine(
            image,
            M,
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        return compensated_image

    def _log_all_camera_parameters(self) -> None:
        """
        Read and log all camera parameters from SDK.
        This helps diagnose parameter differences between cameras.
        """
        if not self.cam:
            logger.warning("Cannot log camera parameters: camera not initialized")
            return

        logger.info("=" * 80)
        logger.info(f"=== All Camera Parameters for {self.camera_manufacturer} {self.camera_model} (SN: {self.camera_serial}) ===")
        logger.info("=" * 80)

        # List of common camera parameters to read
        # Float parameters
        float_params = [
            "ExposureTime",
            "Gain",
            "Gamma",
            "BalanceWhiteRed",
            "BalanceWhiteGreen",
            "BalanceWhiteBlue",
            "AcquisitionFrameRate",
            "ResultingFrameRate",
            "BlackLevel",
            "DefectPixelListSize",
            "LineRate",
            "OffsetX",
            "OffsetY",
            "Width",
            "Height",
            "Brightness",
            "Contrast",
            "Saturation",
            "Sharpness",
            "Hue",
        ]

        # Bool parameters
        bool_params = [
            "GammaEnable",
            "AcquisitionFrameRateEnable",
            "BalanceWhiteAuto",
            "GainAuto",
            "ExposureAuto",
            "TriggerMode",
            "TriggerActivation",
        ]

        # Enum parameters
        enum_params = [
            "PixelFormat",
            "TriggerMode",
            "TriggerSource",
            "BalanceWhiteAuto",
            "ExposureAuto",
            "GainAuto",
            "AcquisitionMode",
            "GevSCPD",
            "GammaSelector",
        ]

        # Int parameters (if needed)
        int_params = [
            "WidthMax",
            "HeightMax",
            "PayloadSize",
        ]

        # String parameters
        string_params = [
            "DeviceUserID",
            "DeviceSerialNumber",
            "DeviceModelName",
            "DeviceManufacturerName",
            "DeviceVersion",
            "DeviceFirmwareVersion",
        ]

        # Read float parameters
        logger.info("\n--- Float Parameters ---")
        for param_name in float_params:
            try:
                float_value = MVCC_FLOATVALUE()
                ret = self.cam.MV_CC_GetFloatValue(param_name, float_value)
                if ret == MV_OK:
                    logger.info(f"  {param_name:30s} = {float_value.fCurValue:12.6f} (Min: {float_value.fMin:.6f}, Max: {float_value.fMax:.6f})")
                else:
                    logger.debug(f"  {param_name:30s} = Not available (error: 0x{ret:x})")
            except Exception as e:
                logger.debug(f"  {param_name:30s} = Error reading: {e}")

        # Read bool parameters
        logger.info("\n--- Bool Parameters ---")
        for param_name in bool_params:
            try:
                bool_value = c_bool()
                ret = self.cam.MV_CC_GetBoolValue(param_name, byref(bool_value))
                if ret == MV_OK:
                    logger.info(f"  {param_name:30s} = {bool_value.value}")
                else:
                    logger.debug(f"  {param_name:30s} = Not available (error: 0x{ret:x})")
            except Exception as e:
                logger.debug(f"  {param_name:30s} = Error reading: {e}")

        # Read enum parameters
        logger.info("\n--- Enum Parameters ---")
        for param_name in enum_params:
            try:
                enum_value = MVCC_ENUMVALUE()
                ret = self.cam.MV_CC_GetEnumValue(param_name, enum_value)
                if ret == MV_OK:
                    logger.info(f"  {param_name:30s} = {enum_value.nCurValue} (supported values: {enum_value.nSupportedNum})")
                else:
                    logger.debug(f"  {param_name:30s} = Not available (error: 0x{ret:x})")
            except Exception as e:
                logger.debug(f"  {param_name:30s} = Error reading: {e}")

        # Read int parameters
        logger.info("\n--- Int Parameters ---")
        for param_name in int_params:
            try:
                int_value = MVCC_INTVALUE()
                ret = self.cam.MV_CC_GetIntValue(param_name, int_value)
                if ret == MV_OK:
                    logger.info(f"  {param_name:30s} = {int_value.nCurValue} (Min: {int_value.nMin}, Max: {int_value.nMax})")
                else:
                    logger.debug(f"  {param_name:30s} = Not available (error: 0x{ret:x})")
            except Exception as e:
                logger.debug(f"  {param_name:30s} = Error reading: {e}")

        # Read string parameters
        logger.info("\n--- String Parameters ---")
        for param_name in string_params:
            try:
                string_value = MVCC_STRINGVALUE()
                ret = self.cam.MV_CC_GetStringValue(param_name, string_value)
                if ret == MV_OK:
                    value_str = string_value.chCurValue.decode('utf-8', errors='ignore').rstrip('\x00')
                    logger.info(f"  {param_name:30s} = {value_str}")
                else:
                    logger.debug(f"  {param_name:30s} = Not available (error: 0x{ret:x})")
            except Exception as e:
                logger.debug(f"  {param_name:30s} = Error reading: {e}")

        logger.info("=" * 80)
        logger.info("")

    def _initialize_camera(self) -> bool:
        """Initialize Hikvision camera and configure settings."""
        try:
            # Initialize SDK
            ret = MvCamera.MV_CC_Initialize()
            if ret != MV_OK:
                logger.error(f"Failed to initialize Hikvision SDK, error code: 0x{ret:x}")
                return False

            # Enumerate devices
            device_list = MV_CC_DEVICE_INFO_LIST()
            device_list.nDeviceNum = 0

            tlayerType = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE |
                         MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE | MV_GENTL_GIGE_DEVICE)

            ret = MvCamera.MV_CC_EnumDevices(tlayerType, device_list)
            if ret != MV_OK:
                logger.error(f"Failed to enumerate devices, error code: 0x{ret:x}")
                MvCamera.MV_CC_Finalize()
                return False

            if device_list.nDeviceNum == 0:
                logger.error("No Hikvision cameras found!")
                MvCamera.MV_CC_Finalize()
                return False

            logger.info(f"Found {device_list.nDeviceNum} Hikvision camera(s)")

            # Physical selection is independent of the logical SHM ID.
            if self.capture_id < 0 or self.capture_id >= device_list.nDeviceNum:
                logger.error(f"Capture index {self.capture_id} out of range (found {device_list.nDeviceNum} cameras)")
                MvCamera.MV_CC_Finalize()
                return False

            # Get device info
            stDeviceList = cast(device_list.pDeviceInfo[self.capture_id], POINTER(MV_CC_DEVICE_INFO)).contents

            # Extract camera information
            if stDeviceList.nTLayerType == MV_USB_DEVICE:
                usb_info = stDeviceList.SpecialInfo.stUsb3VInfo
                self.camera_model = ''.join([chr(c) for c in usb_info.chModelName if c != 0])
                self.camera_serial = ''.join([chr(c) for c in usb_info.chSerialNumber if c != 0])
                self.camera_manufacturer = ''.join([chr(c) for c in usb_info.chManufacturerName if c != 0])
            else:
                gige_info = stDeviceList.SpecialInfo.stGigEInfo
                self.camera_model = ''.join([chr(c) for c in gige_info.chModelName if c != 0])
                self.camera_serial = ''.join([chr(c) for c in gige_info.chSerialNumber if c != 0])
                self.camera_manufacturer = ''.join([chr(c) for c in gige_info.chManufacturerName if c != 0])

            logger.info(f"Selected camera: {self.camera_manufacturer} {self.camera_model} (SN: {self.camera_serial})")

            # Load circle calibration file for this camera (optional)
            if self.load_calibration:
                self._load_circle_calibration()
            else:
                logger.info("Calibration loading disabled, offset compensation will be skipped")
                self.circle_calibration = None

            # Create camera instance
            self.cam = MvCamera()

            # Create handle
            ret = self.cam.MV_CC_CreateHandle(stDeviceList)
            if ret != MV_OK:
                logger.error(f"Failed to create handle, error code: 0x{ret:x}")
                MvCamera.MV_CC_Finalize()
                return False

            # Open device
            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != MV_OK:
                logger.error(f"Failed to open device, error code: 0x{ret:x}")
                self.cam.MV_CC_DestroyHandle()
                MvCamera.MV_CC_Finalize()
                return False

            # Configure camera parameters (must be done after opening device, before starting grabbing)
            if not self._set_camera_parameters():
                logger.warning("Failed to configure camera parameters, continuing with defaults...")

            # Set trigger mode to off (free run mode)
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if ret != MV_OK:
                logger.warning(f"Failed to set trigger mode, error code: 0x{ret:x}")

            # Increase SDK buffer size for async mode to prevent frame drops
            # Default is 1, increase to 20 to buffer more frames in async mode
            try:
                ret = self.cam.MV_CC_SetImageNodeNum(1000)
                if ret == MV_OK:
                    logger.info("Set SDK image buffer count to 1000 for async mode")
                else:
                    logger.warning(f"Failed to set SDK buffer count, error code: 0x{ret:x}")
            except Exception as e:
                logger.warning(f"Failed to set SDK buffer count: {e}")

            # Log all camera parameters for debugging
            self._log_all_camera_parameters()

            logger.info(f"Hikvision camera {self.camera_serial} initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if self.cam:
                try:
                    self.cam.MV_CC_CloseDevice()
                    self.cam.MV_CC_DestroyHandle()
                except Exception:
                    pass
            try:
                MvCamera.MV_CC_Finalize()
            except Exception:
                pass
            return False

    def _capture_frame(self) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """
        Capture a frame from the Hikvision camera.

        Returns:
            tuple: (timestamp_ns, numpy.ndarray) or (None, None) if capture failed
        """
        if not self.cam:
            return None, None

        try:
            # Get one frame with timeout (increase to 2000ms for async mode)
            stFrameInfo = MV_FRAME_OUT()
            ret = self.cam.MV_CC_GetImageBuffer(stFrameInfo, 2000)  # 2000ms timeout

            if ret != MV_OK:
                if ret != 0x80000007:  # MV_E_NODATA
                    logger.warning(f"Failed to get image buffer, error code: 0x{ret:x}")
                return None, None

            # Get timestamp
            timestamp_ns = time.time_ns()

            try:
                # Validate frame length before processing
                frame_len = stFrameInfo.stFrameInfo.nFrameLen
                height = stFrameInfo.stFrameInfo.nHeight
                width = stFrameInfo.stFrameInfo.nWidth
                pixel_type = stFrameInfo.stFrameInfo.enPixelType

                # Validate frame size is reasonable
                if frame_len == 0:
                    logger.warning("Received frame with zero length, skipping")
                    self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                    return None, None

                # Calculate expected frame size based on pixel format
                expected_size = 0
                if pixel_type == PixelType_Gvsp_Mono8:
                    expected_size = height * width
                elif pixel_type in [PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed]:
                    expected_size = height * width * 3
                elif pixel_type in [PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8,
                                   PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8]:
                    expected_size = height * width
                else:
                    logger.warning(f"Unknown pixel format: 0x{pixel_type:x}, using frame_len={frame_len}")
                    expected_size = frame_len

                # Validate frame_len matches expected size
                if expected_size > 0 and frame_len != expected_size:
                    logger.error(f"Frame size mismatch: frame_len={frame_len}, expected={expected_size} "
                               f"for {height}x{width} with pixel_type=0x{pixel_type:x}")
                    self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                    return None, None

                # Convert image data to numpy array
                image_data = cast(stFrameInfo.pBufAddr, POINTER(c_ubyte * frame_len)).contents
                image = np.frombuffer(image_data, dtype=np.uint8)

                # Validate image buffer size matches frame_len
                if len(image) != frame_len:
                    logger.error(f"Image buffer size mismatch: expected {frame_len}, got {len(image)}")
                    self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                    return None, None

                # Process based on pixel format (already extracted above)
                if pixel_type == PixelType_Gvsp_Mono8:
                    # Mono8 format: 1 byte per pixel
                    # Double-check size before reshape (already validated above, but be extra safe)
                    if len(image) != height * width:
                        logger.error(f"Mono8 frame size mismatch before reshape: got {len(image)} bytes, expected {height * width}")
                        self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                        return None, None
                    image = image.reshape((height, width))
                    if self.is_color:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                elif pixel_type == PixelType_Gvsp_RGB8_Packed:
                    # RGB8 format: 3 bytes per pixel
                    # Double-check size before reshape
                    if len(image) != height * width * 3:
                        logger.error(f"RGB8 frame size mismatch before reshape: got {len(image)} bytes, expected {height * width * 3}")
                        self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                        return None, None
                    image = image.reshape((height, width, 3))
                    # Already in RGB format

                elif pixel_type == PixelType_Gvsp_BGR8_Packed:
                    # BGR8 format: 3 bytes per pixel
                    # Double-check size before reshape
                    if len(image) != height * width * 3:
                        logger.error(f"BGR8 frame size mismatch before reshape: got {len(image)} bytes, expected {height * width * 3}")
                        self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                        return None, None
                    image = image.reshape((height, width, 3))
                    if self.is_color:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                elif pixel_type in [PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8,
                                   PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8]:
                    # Bayer format: 1 byte per pixel (needs demosaicing)
                    # Double-check size before reshape
                    if len(image) != height * width:
                        logger.error(f"Bayer frame size mismatch before reshape: got {len(image)} bytes, expected {height * width}")
                        self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                        return None, None
                    image = image.reshape((height, width))
                    if pixel_type == PixelType_Gvsp_BayerGR8:
                        image = cv2.cvtColor(image, cv2.COLOR_BayerGR2RGB if self.is_color else cv2.COLOR_BayerGR2GRAY)
                    elif pixel_type == PixelType_Gvsp_BayerRG8:
                        image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB if self.is_color else cv2.COLOR_BayerRG2GRAY)
                    elif pixel_type == PixelType_Gvsp_BayerGB8:
                        image = cv2.cvtColor(image, cv2.COLOR_BayerGB2RGB if self.is_color else cv2.COLOR_BayerGB2GRAY)
                    elif pixel_type == PixelType_Gvsp_BayerBG8:
                        image = cv2.cvtColor(image, cv2.COLOR_BayerBG2RGB if self.is_color else cv2.COLOR_BayerBG2GRAY)
                else:
                    logger.warning(f"Unsupported pixel format: 0x{pixel_type:x}")
                    self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                    return None, None

                # Apply configured rotation before resizing
                image = self._apply_rotation(image)

                # Resize if necessary
                target_shape = self.data_shape
                if self.is_color:
                    target_height, target_width, _ = target_shape
                    if image.shape[:2] != (target_height, target_width):
                        if not self._resize_warned:
                            logger.warning(f"Resizing frame from {image.shape[:2]} to ({target_height}, {target_width})")
                            self._resize_warned = True
                        image = cv2.resize(image, (target_width, target_height))
                else:
                    target_height, target_width = target_shape
                    if image.shape[:2] != (target_height, target_width):
                        if not self._resize_warned:
                            logger.warning(f"Resizing frame from {image.shape[:2]} to ({target_height}, {target_width})")
                            self._resize_warned = True
                        image = cv2.resize(image, (target_width, target_height))

                # Convert to required data type
                image = image.astype(self.data_dtype)

                # Apply offset compensation based on circle calibration
                image = self._apply_offset_compensation(image)

                # Free image buffer
                self.cam.MV_CC_FreeImageBuffer(stFrameInfo)

                return timestamp_ns, image

            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                logger.error(f"  Frame info: frame_len={frame_len}, height={height}, width={width}, pixel_type=0x{pixel_type:x}")
                logger.error(f"  Image buffer: len={len(image) if 'image' in locals() else 'N/A'}, expected={expected_size if 'expected_size' in locals() else 'N/A'}")
                self.cam.MV_CC_FreeImageBuffer(stFrameInfo)
                return None, None

        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None

    def _cleanup_camera(self) -> None:
        """Clean up camera resources."""
        if self.cam:
            try:
                # Stop grabbing if it's running
                self.cam.MV_CC_StopGrabbing()
            except Exception:
                pass

            try:
                # Close device
                self.cam.MV_CC_CloseDevice()
            except Exception:
                pass

            try:
                # Destroy handle
                self.cam.MV_CC_DestroyHandle()
            except Exception:
                pass

            self.cam = None

            try:
                # Finalize SDK
                MvCamera.MV_CC_Finalize()
            except Exception:
                pass

            logger.info("Hikvision camera released")

    def start_server(self) -> None:
        """Start the camera device server and run frame capture loop."""
        if self.running:
            logger.info(f"Hikvision camera {self.camera_serial} is already running")
            return

        logger.info(f"Starting Hikvision camera server {self.camera_serial}...")
        self.running = True

        # Create shared memory
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")

        # Start grabbing
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != MV_OK:
            logger.error(f"Failed to start grabbing, error code: 0x{ret:x}")
            self.running = False
            self._cleanup_shared_memory()
            raise RuntimeError("Failed to start grabbing")

        logger.info(f"Hikvision camera server started. Shared memory: {self.shared_memory_name}")
        logger.info(f"Buffer configuration: {self.buffer_size} frames, {self.frame_size:,} bytes per frame")

        # Create precise timing function
        wait_for_next_iteration = precise_loop_timing(self.update_interval)

        # Main frame capture loop
        while self.running:
            try:
                timestamp_ns, frame_array = self._capture_frame()
                if frame_array is not None:
                    self._write_array_to_shm_with_timestamp(frame_array, timestamp_ns)

                # Wait for next iteration using precise timing
                #advice: Set hardware frequency via Hikvision software, comment out the below wait statement.
                wait_for_next_iteration()
            except Exception as e:
                logger.error(f"Error in frame capture: {e}")
                break

    def stop_server(self) -> None:
        """Stop the camera device server."""
        if not self.running:
            return

        logger.info(f"Stopping Hikvision camera server {self.camera_serial}...")
        self.running = False
        self._cleanup_shared_memory()
        logger.info("Hikvision camera server stopped")

    def close(self) -> None:
        """Close the camera device and release all resources."""
        self.stop_server()
        self._cleanup_camera()
        logger.info(f"Hikvision camera {self.camera_serial} closed")

    def __del__(self) -> None:
        """Destructor to ensure camera and shared memory resources are released."""
        try:
            self._cleanup_camera()
            if hasattr(self, 'shared_memory') and self.shared_memory:
                self._cleanup_shared_memory()
        except Exception:
            pass

    def get_device_info(self) -> Dict[str, Any]:
        """Get camera device information."""
        info = super().get_device_info()
        info.update({
            "camera_type": "Hikvision",
            "camera_model": self.camera_model,
            "camera_serial": self.camera_serial,
            "camera_manufacturer": self.camera_manufacturer,
            "is_color": self.is_color,
            "camera_opened": self.cam is not None,
            "load_calibration": self.load_calibration,
            "calibration_loaded": self.circle_calibration is not None,
            "hardware_parameters": {
                "pixel_format": self.pixel_format,
                "frame_rate_enable": self.frame_rate_enable,
                "exposure_time": self.exposure_time,
                "gain": self.gain,
                "gamma_enable": self.gamma_enable,
                "gamma": self.gamma,
                "rotation_angle": self.rotation_angle
            }
        })
        return info
