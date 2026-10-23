#!/usr/bin/env python3
"""
Hikvision Camera Scanner - Scan for available Hikvision cameras and get device information.

Author: Assistant
"""

import sys
import os
import argparse
from typing import List, Dict, Any, Optional
from ctypes import c_uint, c_ubyte, Structure, Union, POINTER, byref, create_string_buffer, cast

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger_config import logger

# Import Hikvision camera modules
try:
    from third_party.MvImport.MvCameraControl_class import MvCamera
    from third_party.MvImport.CameraParams_header import MV_CC_DEVICE_INFO, MV_CC_DEVICE_INFO_LIST, MV_GIGE_DEVICE_INFO, MV_USB3_DEVICE_INFO
    from third_party.MvImport.MvErrorDefine_const import MV_OK
    from third_party.MvImport.CameraParams_const import MV_GIGE_DEVICE, MV_USB_DEVICE, MV_GENTL_CAMERALINK_DEVICE, MV_GENTL_CXP_DEVICE, MV_GENTL_XOF_DEVICE, MV_GENTL_GIGE_DEVICE
except ImportError as e:
    logger.error(f"Failed to import Hikvision camera modules: {e}")
    logger.error("Please ensure MvImport directory is properly set up")
    sys.exit(1)


def get_device_info_string(info_bytes: bytes) -> str:
    """Convert device info bytes to string, removing null terminators."""
    try:
        return info_bytes.decode('utf-8').rstrip('\x00')
    except UnicodeDecodeError:
        return info_bytes.hex()


def scan_hik_cameras(verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Scan for available Hikvision cameras and get device information.
    
    Args:
        verbose: Whether to show scanning progress
        
    Returns:
        List of available camera information dictionaries
    """
    available_cameras = []
    
    logger.info("Scanning for Hikvision cameras...")
    if verbose:
        logger.info("-" * 40)
    
    try:
        # Initialize SDK first
        ret = MvCamera.MV_CC_Initialize()
        if ret != MV_OK:
            logger.error(f"Failed to initialize SDK, error code: {ret}")
            return available_cameras
        
        # Create device info list - exactly like in the example
        device_list = MV_CC_DEVICE_INFO_LIST()
        device_list.nDeviceNum = 0
        
        # Enumerate devices - use the same layer types as in the example
        tlayerType = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE | 
                     MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE | MV_GENTL_GIGE_DEVICE)
        
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, device_list)
        
        if ret != MV_OK:
            logger.error(f"Failed to enumerate devices, error code: {ret}")
            MvCamera.MV_CC_Finalize()
            return available_cameras
        
        device_count = device_list.nDeviceNum
        logger.info(f"Found {device_count} Hikvision camera(s)")
        
        if device_count == 0:
            logger.warning("No Hikvision cameras found!")
            MvCamera.MV_CC_Finalize()
            return available_cameras
        
        # Process each device
        for i in range(device_count):
            # Get device info using cast like in the example
            mvcc_dev_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            
            camera_info = {
                'device_index': i,
                'transport_layer': mvcc_dev_info.nTLayerType,
                'device_type': mvcc_dev_info.nDevTypeInfo,
                'mac_address_high': mvcc_dev_info.nMacAddrHigh,
                'mac_address_low': mvcc_dev_info.nMacAddrLow,
                'major_version': mvcc_dev_info.nMajorVer,
                'minor_version': mvcc_dev_info.nMinorVer
            }
            
            # Get device-specific information based on transport layer
            if mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                # USB device
                usb_info = mvcc_dev_info.SpecialInfo.stUsb3VInfo
                
                # Extract string information using the same method as in the example
                manufacturer_name = ''.join([chr(c) for c in usb_info.chManufacturerName if c != 0])
                model_name = ''.join([chr(c) for c in usb_info.chModelName if c != 0])
                device_version = ''.join([chr(c) for c in usb_info.chDeviceVersion if c != 0])
                serial_number = ''.join([chr(c) for c in usb_info.chSerialNumber if c != 0])
                user_defined_name = ''.join([chr(c) for c in usb_info.chUserDefinedName if c != 0])
                vendor_name = ''.join([chr(c) for c in usb_info.chVendorName if c != 0])
                family_name = ''.join([chr(c) for c in usb_info.chFamilyName if c != 0])
                
                camera_info.update({
                    'connection_type': 'USB',
                    'manufacturer_name': manufacturer_name,
                    'model_name': model_name,
                    'device_version': device_version,
                    'serial_number': serial_number,
                    'user_defined_name': user_defined_name,
                    'vendor_name': vendor_name,
                    'family_name': family_name,
                    'device_number': usb_info.nDeviceNumber,
                    'device_guid': usb_info.chDeviceGUID,
                    'device_address': usb_info.nDeviceAddress,
                    'vendor_id': usb_info.idVendor,
                    'product_id': usb_info.idProduct,
                    'usb_protocol': usb_info.nbcdUSB
                })
                
            elif (mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE or 
                  mvcc_dev_info.nTLayerType == MV_GENTL_GIGE_DEVICE):
                # GigE device
                gige_info = mvcc_dev_info.SpecialInfo.stGigEInfo
                
                # Extract string information using the same method as in the example
                manufacturer_name = ''.join([chr(c) for c in gige_info.chManufacturerName if c != 0])
                model_name = ''.join([chr(c) for c in gige_info.chModelName if c != 0])
                device_version = ''.join([chr(c) for c in gige_info.chDeviceVersion if c != 0])
                serial_number = ''.join([chr(c) for c in gige_info.chSerialNumber if c != 0])
                user_defined_name = ''.join([chr(c) for c in gige_info.chUserDefinedName if c != 0])
                manufacturer_specific_info = ''.join([chr(c) for c in gige_info.chManufacturerSpecificInfo if c != 0])
                
                # Extract IP address like in the example
                nip1 = ((gige_info.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((gige_info.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((gige_info.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (gige_info.nCurrentIp & 0x000000ff)
                ip_address_str = f"{nip1}.{nip2}.{nip3}.{nip4}"
                
                camera_info.update({
                    'connection_type': 'GigE',
                    'ip_address': gige_info.nCurrentIp,
                    'ip_address_str': ip_address_str,
                    'subnet_mask': gige_info.nCurrentSubNetMask,
                    'default_gateway': gige_info.nDefultGateWay,
                    'manufacturer_name': manufacturer_name,
                    'model_name': model_name,
                    'device_version': device_version,
                    'serial_number': serial_number,
                    'user_defined_name': user_defined_name,
                    'manufacturer_specific_info': manufacturer_specific_info,
                    'ip_config_option': gige_info.nIpCfgOption,
                    'ip_config_current': gige_info.nIpCfgCurrent,
                    'network_export': gige_info.nNetExport
                })
            
            available_cameras.append(camera_info)
            
            if verbose:
                logger.info(f"Camera {i}: {camera_info.get('manufacturer_name', 'Unknown')} "
                          f"{camera_info.get('model_name', 'Unknown')} "
                          f"(SN: {camera_info.get('serial_number', 'Unknown')})")
        
        # Clean up
        MvCamera.MV_CC_Finalize()
        
    except Exception as e:
        logger.error(f"Error scanning Hikvision cameras: {e}")
        if verbose:
            import traceback
            logger.error(traceback.format_exc())
        try:
            MvCamera.MV_CC_Finalize()
        except:
            pass
    
    return available_cameras


def print_camera_summary(cameras: List[Dict[str, Any]]) -> None:
    """Print a summary of all found Hikvision cameras."""
    if not cameras:
        logger.warning("No Hikvision cameras found!")
        return
    
    logger.info(f"\nFound {len(cameras)} Hikvision camera(s):")
    logger.info("=" * 80)
    
    for i, cam in enumerate(cameras):
        logger.info(f"Camera {i}:")
        logger.info(f"  Connection Type: {cam.get('connection_type', 'Unknown')}")
        logger.info(f"  Manufacturer: {cam.get('manufacturer_name', 'Unknown')}")
        logger.info(f"  Model: {cam.get('model_name', 'Unknown')}")
        logger.info(f"  Serial Number: {cam.get('serial_number', 'Unknown')}")
        logger.info(f"  Device Version: {cam.get('device_version', 'Unknown')}")
        logger.info(f"  Transport Layer: 0x{cam.get('transport_layer', 0):x}")
        logger.info(f"  Device Type: 0x{cam.get('device_type', 0):x}")
        
        if cam.get('connection_type') == 'GigE':
            logger.info(f"  IP Address: {cam.get('ip_address_str', 'Unknown')}")
            logger.info(f"  Subnet Mask: 0x{cam.get('subnet_mask', 0):x}")
            logger.info(f"  Default Gateway: 0x{cam.get('default_gateway', 0):x}")
        elif cam.get('connection_type') == 'USB':
            logger.info(f"  Device Number: {cam.get('device_number', 'Unknown')}")
            logger.info(f"  Device Address: {cam.get('device_address', 'Unknown')}")
            logger.info(f"  Vendor ID: 0x{cam.get('vendor_id', 0):x}")
            logger.info(f"  Product ID: 0x{cam.get('product_id', 0):x}")
            logger.info(f"  USB Protocol: 0x{cam.get('usb_protocol', 0):x}")
            if cam.get('vendor_name'):
                logger.info(f"  Vendor Name: {cam.get('vendor_name')}")
            if cam.get('family_name'):
                logger.info(f"  Family Name: {cam.get('family_name')}")
        
        if cam.get('user_defined_name'):
            logger.info(f"  User Defined Name: {cam.get('user_defined_name')}")
        
        if cam.get('manufacturer_specific_info'):
            logger.info(f"  Manufacturer Specific Info: {cam.get('manufacturer_specific_info')}")
        
        logger.info("")


def main() -> None:
    """Main function to run the Hikvision camera scanner."""
    parser = argparse.ArgumentParser(description="Scan for Hikvision cameras and get device information")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed scanning progress")
    
    args = parser.parse_args()
    
    logger.info("Hikvision Camera Scanner")
    logger.info("=======================")
    
    # Scan cameras
    cameras = scan_hik_cameras(verbose=args.verbose)
    
    # Print results
    print_camera_summary(cameras)
    
    logger.info(f"Found {len(cameras)} Hikvision camera(s)")


if __name__ == "__main__":
    main()
