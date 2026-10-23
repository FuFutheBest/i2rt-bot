'''
Hikvision Industrial Camera Latency Test Script
Adapted from USB camera version for Hikvision industrial cameras
Author: Jun Lv, Han Xue, Yesen Chen
'''
import sys
import cv2
import qrcode
import time
import numpy as np
from collections import deque
from tqdm import tqdm
from matplotlib import pyplot as plt
from threading import Thread, Event, Lock
import argparse
import json
from datetime import datetime
import os
from ctypes import *
import ctypes

sys.path.append("../../MvImport")
sys.path.append("/home/yesenchen/hik/Python/MvImport")
from MvCameraControl_class import *

# Constants for latency compensation
DISPLAY_LATENCY = 0.016  # Typical display latency in seconds (16ms for 60Hz display, 7ms for 144Hz)
QR_GENERATION_OVERHEAD = 0.001  # Estimated QR code generation overhead

# Define callback function type
winfun_ctype = CFUNCTYPE
FrameInfoCallBack = winfun_ctype(None, POINTER(MV_FRAME_OUT), c_void_p, c_bool)


class HikCamera:
    """Hikvision industrial camera wrapper"""
    
    def __init__(self, camera_index: int = 0, width: int = 1920, height: int = 1200, fps: int = 60, exposure_time: int = 5000):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.exposure_time = exposure_time  # Exposure time in microseconds
        self.cam = None
        self.frame_data = None
        self.frame_lock = Lock()
        self.CALL_BACK_FUN = None
        
    def image_callback(self, pstFrame, pUser, bAutoFree):
        """Image callback function for receiving frames from camera"""
        try:
            stFrame = cast(pstFrame, POINTER(MV_FRAME_OUT)).contents
            if not stFrame:
                return

            # Convert image data to numpy array
            image_data = ctypes.cast(
                stFrame.pBufAddr,
                ctypes.POINTER(ctypes.c_ubyte * stFrame.stFrameInfo.nFrameLen)
            ).contents
            image = np.frombuffer(image_data, dtype=np.uint8)

            # Process image based on pixel format
            if stFrame.stFrameInfo.enPixelType == PixelType_Gvsp_Mono8:
                image = image.reshape((stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth))
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif stFrame.stFrameInfo.enPixelType == PixelType_Gvsp_RGB8_Packed:
                image = image.reshape((stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth, 3))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif stFrame.stFrameInfo.enPixelType == PixelType_Gvsp_BGR8_Packed:
                image = image.reshape((stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth, 3))
            elif stFrame.stFrameInfo.enPixelType in [
                PixelType_Gvsp_BayerGR8,
                PixelType_Gvsp_BayerRG8,
                PixelType_Gvsp_BayerGB8,
                PixelType_Gvsp_BayerBG8,
            ]:
                image = image.reshape((stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth))
                if stFrame.stFrameInfo.enPixelType == PixelType_Gvsp_BayerGR8:
                    image = cv2.cvtColor(image, cv2.COLOR_BayerGR2BGR)
                elif stFrame.stFrameInfo.enPixelType == PixelType_Gvsp_BayerRG8:
                    image = cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)
                elif stFrame.stFrameInfo.enPixelType == PixelType_Gvsp_BayerGB8:
                    image = cv2.cvtColor(image, cv2.COLOR_BayerGB2BGR)
                elif stFrame.stFrameInfo.enPixelType == PixelType_Gvsp_BayerBG8:
                    image = cv2.cvtColor(image, cv2.COLOR_BayerBG2BGR)

            with self.frame_lock:
                self.frame_data = image.copy()

        except Exception as e:
            print(f"Error in image callback: {e}")
        
    def start(self):
        """Start camera capture"""
        # Initialize SDK
        MvCamera.MV_CC_Initialize()
        
        # Enumerate devices
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE | 
                     MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
        
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            raise RuntimeError(f"Failed to enumerate devices! ret[0x{ret:x}]")
        
        if deviceList.nDeviceNum == 0:
            raise RuntimeError("No devices found!")
        
        print(f"Found {deviceList.nDeviceNum} device(s)")
        
        # Display device information
        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print(f"\nUSB Device: [{i}]")
                strModeName = ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName if c != 0])
                print(f"Model: {strModeName}")
                strSerialNumber = ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber if c != 0])
                print(f"Serial Number: {strSerialNumber}")
            elif mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE or mvcc_dev_info.nTLayerType == MV_GENTL_GIGE_DEVICE:
                print(f"\nGigE Device: [{i}]")
                strModeName = ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName if c != 0])
                print(f"Model: {strModeName}")
                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print(f"Current IP: {nip1}.{nip2}.{nip3}.{nip4}")
        
        # Select device (use camera_index parameter)
        if self.camera_index >= deviceList.nDeviceNum:
            raise RuntimeError(f"Camera index {self.camera_index} out of range!")
        
        print(f"\nConnecting to device [{self.camera_index}]...")
        
        # Create camera instance
        self.cam = MvCamera()
        
        # Select device and create handle
        stDeviceList = cast(deviceList.pDeviceInfo[self.camera_index], POINTER(MV_CC_DEVICE_INFO)).contents
        
        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            raise RuntimeError(f"Failed to create handle! ret[0x{ret:x}]")
        
        # Open device
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise RuntimeError(f"Failed to open device! ret[0x{ret:x}]")
        
        # Set trigger mode to off
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print(f"Warning: Failed to set trigger mode! ret[0x{ret:x}]")
        
        # Set frame rate (if supported)
        ret = self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        if ret == 0:
            ret = self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.fps))
            if ret != 0:
                print(f"Warning: Failed to set frame rate! ret[0x{ret:x}]")
        
        # Set exposure time (manual mode)
        ret = self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)  # 0 = Off (manual mode)
        if ret == 0:
            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(self.exposure_time))
            if ret != 0:
                print(f"Warning: Failed to set exposure time! ret[0x{ret:x}]")
            else:
                print(f"Set exposure time to {self.exposure_time} μs")
        
        # Register image callback
        self.CALL_BACK_FUN = FrameInfoCallBack(self.image_callback)
        ret = self.cam.MV_CC_RegisterImageCallBackEx2(self.CALL_BACK_FUN, ctypes.py_object(self), True)
        if ret != 0:
            raise RuntimeError(f"Failed to register image callback! ret[0x{ret:x}]")
        
        # Start grabbing
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"Failed to start grabbing! ret[0x{ret:x}]")
        
        print("Camera started successfully!")
        
    def get_rgb_frame(self):
        """Get RGB frame from camera"""
        with self.frame_lock:
            if self.frame_data is not None:
                return self.frame_data.copy()
        return None
        
    def stop(self):
        """Stop camera capture"""
        if self.cam:
            # Stop grabbing
            ret = self.cam.MV_CC_StopGrabbing()
            if ret != 0:
                print(f"Warning: Failed to stop grabbing! ret[0x{ret:x}]")
            
            # Close device
            ret = self.cam.MV_CC_CloseDevice()
            if ret != 0:
                print(f"Warning: Failed to close device! ret[0x{ret:x}]")
            
            # Destroy handle
            ret = self.cam.MV_CC_DestroyHandle()
            if ret != 0:
                print(f"Warning: Failed to destroy handle! ret[0x{ret:x}]")
            
            self.cam = None
        
        # Finalize SDK
        MvCamera.MV_CC_Finalize()
        print("Camera stopped")


def calculate_and_visualize_latency_stats(qr_det_queue, qr_latency_deque, qr_generation_times, 
                                        ts_recv_deque, cam_img_deque, args):
    """Calculate latency statistics and create visualization plots"""
    
    # Convert deques to numpy arrays for easier processing
    raw_latencies = np.array([x for x in qr_det_queue if not np.isnan(x)])
    processing_latencies = np.array(qr_latency_deque)
    qr_gen_times = np.array(qr_generation_times)
    
    if len(raw_latencies) == 0:
        print("No valid latency measurements found!")
        return
    
    # Calculate compensated latencies
    avg_processing_latency = np.mean(processing_latencies) if len(processing_latencies) > 0 else 0
    avg_qr_gen_time = np.mean(qr_gen_times) if len(qr_gen_times) > 0 else QR_GENERATION_OVERHEAD
    
    compensated_latencies = raw_latencies - avg_processing_latency - avg_qr_gen_time - DISPLAY_LATENCY
    
    # Calculate statistics
    stats = {
        'raw_latency': {
            'mean': float(np.mean(raw_latencies)),
            'std': float(np.std(raw_latencies)),
            'var': float(np.var(raw_latencies)),
            'min': float(np.min(raw_latencies)),
            'max': float(np.max(raw_latencies)),
            'median': float(np.median(raw_latencies)),
            'count': len(raw_latencies)
        },
        'compensated_latency': {
            'mean': float(np.mean(compensated_latencies)),
            'std': float(np.std(compensated_latencies)),
            'var': float(np.var(compensated_latencies)),
            'min': float(np.min(compensated_latencies)),
            'max': float(np.max(compensated_latencies)),
            'median': float(np.median(compensated_latencies)),
            'count': len(compensated_latencies)
        },
        'processing_overhead': {
            'qr_generation_avg': float(avg_qr_gen_time),
            'display_processing_avg': float(avg_processing_latency),
            'display_latency_constant': float(DISPLAY_LATENCY)
        },
        'detection_rate': float(len(raw_latencies) / len(qr_det_queue)),
        'test_config': {
            'camera_idx': args.camera_idx,
            'camera_fps': args.camera_fps,
            'display_fps': args.display_fps,
            'qr_size': args.qr_size,
            'n_frames': args.n_frames,
            'exposure_time': args.exposure_time
        }
    }
    
    # Print statistics
    print("\n" + "="*60)
    print("LATENCY ANALYSIS RESULTS")
    print("="*60)
    print(f"Detection Rate: {stats['detection_rate']:.2%} ({stats['raw_latency']['count']}/{len(qr_det_queue)})")
    print(f"\nRaw Latency Statistics:")
    print(f"  Mean: {stats['raw_latency']['mean']*1000:.2f} ms")
    print(f"  Std:  {stats['raw_latency']['std']*1000:.2f} ms")
    print(f"  Var:  {stats['raw_latency']['var']*1000000:.2f} ms²")
    print(f"  Min:  {stats['raw_latency']['min']*1000:.2f} ms")
    print(f"  Max:  {stats['raw_latency']['max']*1000:.2f} ms")
    print(f"  Median: {stats['raw_latency']['median']*1000:.2f} ms")
    
    print(f"\nCompensated Latency Statistics:")
    print(f"  Mean: {stats['compensated_latency']['mean']*1000:.2f} ms")
    print(f"  Std:  {stats['compensated_latency']['std']*1000:.2f} ms")
    print(f"  Var:  {stats['compensated_latency']['var']*1000000:.2f} ms²")
    print(f"  Min:  {stats['compensated_latency']['min']*1000:.2f} ms")
    print(f"  Max:  {stats['compensated_latency']['max']*1000:.2f} ms")
    print(f"  Median: {stats['compensated_latency']['median']*1000:.2f} ms")
    
    print(f"\nProcessing Overhead:")
    print(f"  QR Generation: {stats['processing_overhead']['qr_generation_avg']*1000:.2f} ms")
    print(f"  Display Processing: {stats['processing_overhead']['display_processing_avg']*1000:.2f} ms")
    print(f"  Display Latency: {stats['processing_overhead']['display_latency_constant']*1000:.2f} ms")
    print("="*60)
    
    # Create visualization
    create_latency_plots(raw_latencies, compensated_latencies, stats, args)
    
    # Save results to file
    save_results_to_file(stats, raw_latencies, compensated_latencies)


def create_latency_plots(raw_latencies, compensated_latencies, stats, args):
    """Create comprehensive latency visualization plots"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Hikvision Camera Latency Analysis (Camera {args.camera_idx}, {args.camera_fps}FPS, Exp {args.exposure_time}μs)', 
                 fontsize=16, fontweight='bold')
    
    # Convert to milliseconds for better readability
    raw_ms = raw_latencies * 1000
    comp_ms = compensated_latencies * 1000
    
    # Plot 1: Raw latency histogram
    axes[0, 0].hist(raw_ms, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(stats['raw_latency']['mean']*1000, color='red', linestyle='--', 
                       label=f'Mean: {stats["raw_latency"]["mean"]*1000:.2f}ms')
    axes[0, 0].axvline(stats['raw_latency']['median']*1000, color='green', linestyle='--', 
                       label=f'Median: {stats["raw_latency"]["median"]*1000:.2f}ms')
    axes[0, 0].set_title('Raw Latency Distribution')
    axes[0, 0].set_xlabel('Latency (ms)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Compensated latency histogram
    axes[0, 1].hist(comp_ms, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].axvline(stats['compensated_latency']['mean']*1000, color='red', linestyle='--', 
                       label=f'Mean: {stats["compensated_latency"]["mean"]*1000:.2f}ms')
    axes[0, 1].axvline(stats['compensated_latency']['median']*1000, color='green', linestyle='--', 
                       label=f'Median: {stats["compensated_latency"]["median"]*1000:.2f}ms')
    axes[0, 1].set_title('Compensated Latency Distribution')
    axes[0, 1].set_xlabel('Latency (ms)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Box plot comparison
    box_data = [raw_ms, comp_ms]
    box_labels = ['Raw', 'Compensated']
    bp = axes[0, 2].boxplot(box_data, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[0, 2].set_title('Latency Comparison')
    axes[0, 2].set_ylabel('Latency (ms)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Time series of raw latency
    axes[1, 0].plot(range(len(raw_ms)), raw_ms, 'b-', alpha=0.7, linewidth=1)
    axes[1, 0].axhline(stats['raw_latency']['mean']*1000, color='red', linestyle='--', alpha=0.8)
    axes[1, 0].fill_between(range(len(raw_ms)), 
                           (stats['raw_latency']['mean'] - stats['raw_latency']['std'])*1000,
                           (stats['raw_latency']['mean'] + stats['raw_latency']['std'])*1000,
                           alpha=0.2, color='red', label='±1σ')
    axes[1, 0].set_title('Raw Latency Time Series')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Latency (ms)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Time series of compensated latency
    axes[1, 1].plot(range(len(comp_ms)), comp_ms, 'orange', alpha=0.7, linewidth=1)
    axes[1, 1].axhline(stats['compensated_latency']['mean']*1000, color='red', linestyle='--', alpha=0.8)
    axes[1, 1].fill_between(range(len(comp_ms)), 
                           (stats['compensated_latency']['mean'] - stats['compensated_latency']['std'])*1000,
                           (stats['compensated_latency']['mean'] + stats['compensated_latency']['std'])*1000,
                           alpha=0.2, color='red', label='±1σ')
    axes[1, 1].set_title('Compensated Latency Time Series')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Latency (ms)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Statistics summary table
    axes[1, 2].axis('off')
    
    # Create statistics table
    table_data = [
        ['Metric', 'Raw (ms)', 'Compensated (ms)'],
        ['Mean', f"{stats['raw_latency']['mean']*1000:.2f}", f"{stats['compensated_latency']['mean']*1000:.2f}"],
        ['Std Dev', f"{stats['raw_latency']['std']*1000:.2f}", f"{stats['compensated_latency']['std']*1000:.2f}"],
        ['Variance', f"{stats['raw_latency']['var']*1000000:.2f}", f"{stats['compensated_latency']['var']*1000000:.2f}"],
        ['Min', f"{stats['raw_latency']['min']*1000:.2f}", f"{stats['compensated_latency']['min']*1000:.2f}"],
        ['Max', f"{stats['raw_latency']['max']*1000:.2f}", f"{stats['compensated_latency']['max']*1000:.2f}"],
        ['Median', f"{stats['raw_latency']['median']*1000:.2f}", f"{stats['compensated_latency']['median']*1000:.2f}"],
        ['Samples', f"{stats['raw_latency']['count']}", f"{stats['compensated_latency']['count']}"],
        ['Det. Rate', f"{stats['detection_rate']:.2%}", f"{stats['detection_rate']:.2%}"]
    ]
    
    table = axes[1, 2].table(cellText=table_data[1:], colLabels=table_data[0], 
                            cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(table_data)):
        for j in range(3):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                if j == 0:  # First column
                    table[(i, j)].set_facecolor('#E8F5E8')
                else:
                    table[(i, j)].set_facecolor('#F5F5F5')
    
    axes[1, 2].set_title('Statistics Summary')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"hik_latency_analysis_cam{args.camera_idx}_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {plot_filename}")
    
    plt.show()


def save_results_to_file(stats, raw_latencies, compensated_latencies):
    """Save detailed results to JSON file"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hik_latency_results_{timestamp}.json"
    
    # Prepare data for JSON serialization
    results = {
        'timestamp': datetime.now().isoformat(),
        'statistics': stats,
        'raw_data': {
            'raw_latencies_ms': (raw_latencies * 1000).tolist(),
            'compensated_latencies_ms': (compensated_latencies * 1000).tolist()
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Test Hikvision camera latency using QR codes')
    parser.add_argument('-ci', '--camera_idx', type=int, default=0, help='Camera index')
    parser.add_argument('-qs', '--qr_size', type=int, default=640, help='QR code size')
    parser.add_argument('-cf', '--camera_fps', type=int, default=60, help='Camera FPS')
    parser.add_argument('-sf', '--display_fps', type=int, default=60, help='Display FPS')
    parser.add_argument('-n', '--n_frames', type=int, default=500, help='Number of frames to capture')
    parser.add_argument('-et', '--exposure_time', type=int, default=5000, help='Exposure time in microseconds')
    
    args = parser.parse_args()

    cv2.setNumThreads(1)

    get_max_k = args.n_frames
    cam_img_deque = deque(maxlen=get_max_k)
    ts_recv_deque = deque(maxlen=get_max_k)
    qr_latency_deque = deque(maxlen=get_max_k)
    qr_det_queue = deque(maxlen=get_max_k)

    stop_event = Event()

    cam_img_buffer = deque(maxlen=10*args.camera_fps)
    cam_img_vis_buffer = deque(maxlen=10*args.camera_fps)
    qr_generation_times = deque(maxlen=get_max_k)
    
    # Thread synchronization locks
    cam_buffer_lock = Lock()
    vis_buffer_lock = Lock()
    qr_times_lock = Lock()

    def capture_worker():
        camera = HikCamera(
            camera_index=args.camera_idx,
            width=1920,
            height=1200,
            fps=args.camera_fps,
            exposure_time=args.exposure_time
        )
        camera.start()
        last_print_time = time.time()
        frame_count = 0
        while not stop_event.is_set():
            frame = camera.get_rgb_frame()
            if frame is not None:
                with cam_buffer_lock:
                    cam_img_buffer.append(frame)
                frame_count += 1
                now = time.time()
                if now - last_print_time >= 1.0:
                    elapsed = now - last_print_time
                    fps = frame_count / elapsed
                    print(f"Capture running at {fps:.1f} FPS")
                    frame_count = 0
                    last_print_time = now
            else:
                time.sleep(0.001)
        camera.stop()
        
    capture_thread = Thread(target=capture_worker, daemon=True)
    capture_thread.start()

    def qr_det_worker():
        detector = cv2.QRCodeDetector()
        while not stop_event.is_set():
            with cam_buffer_lock:
                if len(cam_img_buffer) == 0:
                    cam_img = None
                else:
                    cam_img = cam_img_buffer[-1].copy()
            
            if cam_img is None:
                print("Waiting for camera frames...")
                time.sleep(0.1)
                continue
                
            ts_recv = time.time()

            cam_img_deque.append(cam_img.copy())
            ts_recv_deque.append(ts_recv)

            code, corners, _ = detector.detectAndDecodeCurved(cam_img)
            
            # Process QR code detection results
            latency = float('nan')  # Initialize latency
            if len(code) > 0:
                try:
                    ts_qr = float(code)
                    latency = ts_recv - ts_qr
                    qr_det_queue.append(latency)
                except ValueError:
                    qr_det_queue.append(float('nan'))
            else:
                qr_det_queue.append(float('nan'))
                
            # Create visualization image with QR code masking
            cam_img_vis = cam_img.copy()
            
            if corners is not None:
                # Expand the masking area to ensure complete coverage
                corners_expanded = corners.copy()
                center = np.mean(corners_expanded[0], axis=0)
                for i in range(len(corners_expanded[0])):
                    direction = corners_expanded[0][i] - center
                    corners_expanded[0][i] = center + direction * 1.2  # Expand by 20%
                
                # Fill detected QR code area with black to prevent re-detection
                cv2.fillPoly(cam_img_vis, corners_expanded.astype(np.int32), (0, 0, 0))
                
                # Add a colored border to show detection status
                if len(code) > 0:
                    # Green border for successful detection
                    cv2.polylines(cam_img_vis, corners.astype(np.int32), True, (0, 255, 0), 3)
                else:
                    # Red border for failed parsing
                    cv2.polylines(cam_img_vis, corners.astype(np.int32), True, (0, 0, 255), 3)
            
            # Add status text overlay
            status_text = f"Latency: {latency:.3f}s" if len(code) > 0 and not np.isnan(qr_det_queue[-1]) else "No QR detected"
            cv2.putText(cam_img_vis, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add detection rate info
            if len(qr_det_queue) > 10:
                recent_det_rate = 1 - np.mean(np.isnan(list(qr_det_queue)[-20:]))
                det_text = f"Detection Rate: {recent_det_rate:.2f}"
                cv2.putText(cam_img_vis, det_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            with vis_buffer_lock:
                cam_img_vis_buffer.append(cam_img_vis)
            
    qr_det_thread = Thread(target=qr_det_worker, daemon=True)
    qr_det_thread.start()

    print("Press 'c' to capture data, 'q' to quit")
    
    # Initialize compensated_latency variable
    compensated_latency = 0.0
    
    while True:
        t_start = time.time()

        # Measure QR code generation time
        t_qr_gen_start = time.time()
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
        )
        t_sample = time.time()  # Timestamp to embed in QR code
        qr.add_data(str(t_sample))
        qr.make(fit=True)
        pil_img = qr.make_image()
        img = np.array(pil_img).astype(np.uint8) * 255
        img = np.repeat(img[:,:,None], 3, axis=-1)
        img = cv2.resize(img, (args.qr_size, args.qr_size), cv2.INTER_NEAREST)
        t_qr_gen_end = time.time()

        cv2.imshow('Timestamp QRCode', img)
        # Display camera feed with enhanced visualization
        with vis_buffer_lock:
            if len(cam_img_vis_buffer) > 0:
                cam_img_vis = cam_img_vis_buffer[-1].copy()

                # Add real-time statistics overlay
                fps_text = f"Display FPS: {args.display_fps:.1f}"
                cv2.putText(cam_img_vis, fps_text, (10, cam_img_vis.shape[0] - 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                latency_text = f"Camera Latency: {compensated_latency:.3f}s"
                cv2.putText(cam_img_vis, latency_text, (10, cam_img_vis.shape[0] - 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cam_img_vis = cv2.resize(cam_img_vis, (args.qr_size, args.qr_size), cv2.INTER_NEAREST)
                cv2.imshow('Camera Feed', cam_img_vis)
            else:
                # Show placeholder if no camera frames available
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for camera...", (200, 240),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                placeholder = cv2.resize(placeholder, (args.qr_size, args.qr_size), cv2.INTER_NEAREST)
                cv2.imshow('Camera Feed', placeholder)

        t_show = time.time()

        # Store timing information with thread safety
        qr_gen_time = t_qr_gen_end - t_qr_gen_start
        display_processing_time = t_show - t_qr_gen_end

        with qr_times_lock:
            qr_generation_times.append(qr_gen_time)
        qr_latency_deque.append(display_processing_time)

        time.sleep(max(0, 1/args.display_fps - (time.time() - t_start)))

        t_end = time.time()

        # Calculate comprehensive latency including all factors
        raw_latency = np.nanmean(qr_det_queue)
        processing_latency = np.nanmean(qr_latency_deque)

        with qr_times_lock:
            if len(qr_generation_times) > 0:
                avg_qr_gen_time = np.mean(qr_generation_times)
            else:
                avg_qr_gen_time = QR_GENERATION_OVERHEAD

        # Total system latency = measured_latency - processing_delays - display_latency
        compensated_latency = raw_latency - processing_latency - avg_qr_gen_time - DISPLAY_LATENCY

        if len(qr_det_queue) > 0:
            det_rate = 1 - np.mean(np.isnan(qr_det_queue))
        else:
            det_rate = 0.0
            
        print("QR: {:.1f}FPS | Raw: {:.3f}s | Compensated: {:.3f}s | Det: {:.2f} | QRGen: {:.3f}s".format(
            1/(t_end-t_start),
            raw_latency if not np.isnan(raw_latency) else 0,
            compensated_latency if not np.isnan(compensated_latency) else 0,
            det_rate,
            avg_qr_gen_time
        ))

        keycode = cv2.pollKey()
        if keycode == ord('c'):
            print("Capturing data...")
            stop_event.set()
            capture_thread.join()
            qr_det_thread.join()
            break
        elif keycode == ord('q') or keycode == 27:  # 'q' or ESC key
            print("Quitting...")
            stop_event.set()
            capture_thread.join()
            qr_det_thread.join()
            cv2.destroyAllWindows()
            exit(0)
    
    print("Processing captured data...")

    if len(qr_det_queue) == 0:
        print("No QR detection data available!")
        return

    # Calculate latency statistics
    calculate_and_visualize_latency_stats(
        qr_det_queue, qr_latency_deque, qr_generation_times, 
        ts_recv_deque, cam_img_deque, args
    )
    
    # Clean up OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

