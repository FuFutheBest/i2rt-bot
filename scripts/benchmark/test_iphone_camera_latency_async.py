#!/usr/bin/env python3
"""
iPhone Record3D camera latency calibration using timestamp QR codes.

The script displays a QR code that contains the local generation timestamp. The
iPhone observes the screen through Record3D; when the QR is decoded, the camera
frame receive timestamp is compared with the embedded QR timestamp.
"""

import argparse
import json
import time
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import qrcode

try:
    from record3d import Record3DStream
except ImportError as exc:  # pragma: no cover - dependency is hardware-side
    raise SystemExit(
        "record3d package is required. Install/activate the same environment used "
        "by IPhoneCameraDevice."
    ) from exc


DISPLAY_LATENCY_S = 0.016
QR_GENERATION_OVERHEAD_S = 0.001
DEVICE_TYPE_TRUEDEPTH = 0


class IPhoneRecord3DCamera:
    """Small direct Record3D reader for latency calibration."""

    def __init__(
        self,
        *,
        device_index: int = 0,
        device_udid: Optional[str] = None,
        flip_truedepth: bool = True,
        rotation_degree: int = 0,
        output_shape: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        self.device_index = int(device_index)
        self.device_udid = device_udid
        self.flip_truedepth = bool(flip_truedepth)
        self.rotation_degree = self._validate_rotation_degree(rotation_degree)
        self.output_shape = output_shape

        self.session: Optional[Record3DStream] = None
        self._frame_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[Tuple[float, np.ndarray]] = None
        self._connected_info: Dict[str, Any] = {}

    @staticmethod
    def _validate_rotation_degree(rotation_degree: int) -> int:
        rotation_degree = int(rotation_degree)
        if rotation_degree not in (0, 90, 180, 270):
            raise ValueError("rotation_degree must be one of 0, 90, 180, 270")
        return rotation_degree

    def _on_new_frame(self) -> None:
        self._frame_event.set()

    def _on_stream_stopped(self) -> None:
        self._frame_event.set()

    def _select_device(self) -> Any:
        devices = Record3DStream.get_connected_devices()
        if not devices:
            raise RuntimeError(
                "No Record3D devices detected. Connect the iPhone and open Record3D."
            )

        print(f"Found {len(devices)} Record3D device(s):")
        for idx, dev in enumerate(devices):
            print(
                f"  [{idx}] product_id={getattr(dev, 'product_id', None)} "
                f"udid={getattr(dev, 'udid', None)}"
            )

        if self.device_udid:
            for dev in devices:
                if getattr(dev, "udid", None) == self.device_udid:
                    return dev
            raise RuntimeError(f"Record3D device with UDID {self.device_udid} not found")

        if self.device_index >= len(devices):
            raise RuntimeError(
                f"Record3D device index {self.device_index} out of range "
                f"(found {len(devices)} devices)"
            )
        return devices[self.device_index]

    def start(self) -> None:
        dev = self._select_device()
        self.session = Record3DStream()
        self.session.on_new_frame = self._on_new_frame
        self.session.on_stream_stopped = self._on_stream_stopped
        self.session.connect(dev)
        self._connected_info = {
            "product_id": getattr(dev, "product_id", None),
            "udid": getattr(dev, "udid", None),
        }
        print(
            "Connected to Record3D device: "
            f"product_id={self._connected_info['product_id']} "
            f"udid={self._connected_info['udid']}"
        )

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._frame_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.session is not None:
            try:
                self.session.stop_stream()
            except Exception:
                pass
            self.session = None

    def connected_info(self) -> Dict[str, Any]:
        return dict(self._connected_info)

    def get_latest(self) -> Optional[Tuple[float, np.ndarray]]:
        with self._lock:
            if self._latest is None:
                return None
            timestamp_s, frame = self._latest
            return timestamp_s, frame.copy()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        if (
            self.flip_truedepth
            and self.session is not None
            and hasattr(self.session, "get_device_type")
            and self.session.get_device_type() == DEVICE_TYPE_TRUEDEPTH
        ):
            frame = cv2.flip(frame, 1)

        if self.rotation_degree:
            rotate_map = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }
            frame = cv2.rotate(frame, rotate_map[self.rotation_degree])

        if self.output_shape is not None:
            height, width = self.output_shape[:2]
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        return frame.astype(np.uint8, copy=False)

    def _worker(self) -> None:
        frame_count = 0
        last_print = time.time()
        while not self._stop_event.is_set():
            if not self._frame_event.wait(timeout=1.0):
                continue
            self._frame_event.clear()
            if self.session is None:
                continue

            try:
                frame = self.session.get_rgb_frame()
            except Exception as exc:
                print(f"Failed to get iPhone frame: {exc}")
                continue
            if frame is None:
                continue

            receive_time_s = time.time()
            frame = self._process_frame(frame)
            with self._lock:
                self._latest = (receive_time_s, frame.copy())

            frame_count += 1
            now = time.time()
            if now - last_print >= 1.0:
                print(f"iPhone capture running at {frame_count / (now - last_print):.1f} FPS")
                frame_count = 0
                last_print = now


def parse_shape(shape_text: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if not shape_text:
        return None
    values = tuple(int(part.strip()) for part in shape_text.split(",") if part.strip())
    if len(values) not in (2, 3):
        raise ValueError("--shape must look like 480,640 or 480,640,3")
    if len(values) == 2:
        return values[0], values[1], 3
    return values


def make_timestamp_qr(timestamp_s: float, qr_size: int) -> np.ndarray:
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(f"{timestamp_s:.9f}")
    qr.make(fit=True)
    pil_img = qr.make_image()
    img = np.array(pil_img).astype(np.uint8) * 255
    img = np.repeat(img[:, :, None], 3, axis=-1)
    return cv2.resize(img, (qr_size, qr_size), interpolation=cv2.INTER_NEAREST)


def decode_qr(detector: cv2.QRCodeDetector, frame_rgb: np.ndarray) -> Tuple[str, Any]:
    try:
        code, corners, _ = detector.detectAndDecodeCurved(frame_rgb)
    except cv2.error:
        code, corners, _ = detector.detectAndDecode(frame_rgb)
    if not code:
        code, corners, _ = detector.detectAndDecode(frame_rgb)
    return code, corners


def draw_camera_overlay(
    frame_rgb: np.ndarray,
    corners: Any,
    latency_s: float,
    recent_detection_rate: float,
    display_fps: float,
) -> np.ndarray:
    vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if corners is not None:
        try:
            corners_i = corners.astype(np.int32)
            color = (0, 255, 0) if np.isfinite(latency_s) else (0, 0, 255)
            cv2.polylines(vis, corners_i, True, color, 3)
        except Exception:
            pass

    latency_text = (
        f"Latency: {latency_s * 1000:.1f} ms"
        if np.isfinite(latency_s)
        else "No QR detected"
    )
    cv2.putText(vis, latency_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        vis,
        f"Recent detection: {recent_detection_rate:.2%}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        vis,
        f"Display FPS: {display_fps:.1f}",
        (10, vis.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    return vis


def finite_array(values: deque) -> np.ndarray:
    return np.array([value for value in values if np.isfinite(value)], dtype=np.float64)


def stats_for(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "var": float(np.var(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "count": int(values.size),
    }


def build_stats(
    *,
    raw_latencies: np.ndarray,
    compensated_latencies: np.ndarray,
    all_detection_values: deque,
    qr_generation_times: deque,
    display_processing_times: deque,
    args: argparse.Namespace,
    camera_info: Dict[str, Any],
) -> Dict[str, Any]:
    avg_qr_gen = float(np.mean(qr_generation_times)) if qr_generation_times else QR_GENERATION_OVERHEAD_S
    avg_display_processing = (
        float(np.mean(display_processing_times)) if display_processing_times else 0.0
    )
    return {
        "timestamp": datetime.now().isoformat(),
        "camera": {
            "device_index": int(args.device_index),
            "device_udid": args.device_udid,
            **camera_info,
        },
        "raw_latency": stats_for(raw_latencies),
        "compensated_latency": stats_for(compensated_latencies),
        "processing_overhead": {
            "qr_generation_avg_s": avg_qr_gen,
            "display_processing_avg_s": avg_display_processing,
            "display_latency_constant_s": float(args.display_latency_ms / 1000.0),
        },
        "detection_rate": float(raw_latencies.size / max(len(all_detection_values), 1)),
        "detection_attempt_count": int(len(all_detection_values)),
        "recommended_hardware_latency_ms": {
            "mean": float(np.mean(compensated_latencies) * 1000.0),
            "median": float(np.median(compensated_latencies) * 1000.0),
        },
        "test_config": {
            "display_fps": float(args.display_fps),
            "qr_size": int(args.qr_size),
            "n_frames": int(args.n_frames),
            "flip_truedepth": not args.no_flip,
            "rotation_degree": int(args.rotation_degree),
            "shape": args.shape,
        },
    }


def print_stats(stats: Dict[str, Any]) -> None:
    print("\n" + "=" * 68)
    print("IPHONE CAMERA LATENCY ANALYSIS RESULTS")
    print("=" * 68)
    print(
        "Detection Rate: "
        f"{stats['detection_rate']:.2%} "
        f"({stats['raw_latency']['count']}/{stats['detection_attempt_count']})"
    )
    for key, title in (("raw_latency", "Raw Latency"), ("compensated_latency", "Compensated Latency")):
        item = stats[key]
        print(f"\n{title}:")
        print(f"  Mean:   {item['mean'] * 1000:.2f} ms")
        print(f"  Median: {item['median'] * 1000:.2f} ms")
        print(f"  Std:    {item['std'] * 1000:.2f} ms")
        print(f"  Min:    {item['min'] * 1000:.2f} ms")
        print(f"  Max:    {item['max'] * 1000:.2f} ms")
        print(f"  Count:  {item['count']}")

    rec = stats["recommended_hardware_latency_ms"]
    print("\nRecommended hardware_latency_ms:")
    print(f"  mean:   {rec['mean']:.2f}")
    print(f"  median: {rec['median']:.2f}")
    print("=" * 68)


def save_results(
    *,
    stats: Dict[str, Any],
    raw_latencies: np.ndarray,
    compensated_latencies: np.ndarray,
    output_dir: Path,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"iphone_latency_results_{timestamp}.json"
    plot_path = output_dir / f"iphone_latency_analysis_{timestamp}.png"

    payload = {
        "statistics": stats,
        "raw_data": {
            "raw_latencies_ms": (raw_latencies * 1000).tolist(),
            "compensated_latencies_ms": (compensated_latencies * 1000).tolist(),
        },
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    raw_ms = raw_latencies * 1000
    comp_ms = compensated_latencies * 1000
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("iPhone Record3D Camera Latency Analysis", fontsize=16, fontweight="bold")

    axes[0, 0].hist(raw_ms, bins=30, alpha=0.75, color="#2563eb", edgecolor="black")
    axes[0, 0].axvline(np.mean(raw_ms), color="red", linestyle="--", label=f"Mean {np.mean(raw_ms):.1f} ms")
    axes[0, 0].axvline(np.median(raw_ms), color="green", linestyle="--", label=f"Median {np.median(raw_ms):.1f} ms")
    axes[0, 0].set_title("Raw Latency Distribution")
    axes[0, 0].set_xlabel("Latency (ms)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].hist(comp_ms, bins=30, alpha=0.75, color="#ea580c", edgecolor="black")
    axes[0, 1].axvline(np.mean(comp_ms), color="red", linestyle="--", label=f"Mean {np.mean(comp_ms):.1f} ms")
    axes[0, 1].axvline(np.median(comp_ms), color="green", linestyle="--", label=f"Median {np.median(comp_ms):.1f} ms")
    axes[0, 1].set_title("Compensated Latency Distribution")
    axes[0, 1].set_xlabel("Latency (ms)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[0, 2].boxplot([raw_ms, comp_ms], labels=["Raw", "Compensated"], patch_artist=True)
    axes[0, 2].set_title("Latency Comparison")
    axes[0, 2].set_ylabel("Latency (ms)")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(raw_ms, color="#2563eb", linewidth=1)
    axes[1, 0].axhline(np.mean(raw_ms), color="red", linestyle="--")
    axes[1, 0].set_title("Raw Latency Time Series")
    axes[1, 0].set_xlabel("Sample")
    axes[1, 0].set_ylabel("Latency (ms)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(comp_ms, color="#ea580c", linewidth=1)
    axes[1, 1].axhline(np.mean(comp_ms), color="red", linestyle="--")
    axes[1, 1].set_title("Compensated Latency Time Series")
    axes[1, 1].set_xlabel("Sample")
    axes[1, 1].set_ylabel("Latency (ms)")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    rec = stats["recommended_hardware_latency_ms"]
    table_data = [
        ["Metric", "Raw", "Compensated"],
        ["Mean ms", f"{np.mean(raw_ms):.2f}", f"{np.mean(comp_ms):.2f}"],
        ["Median ms", f"{np.median(raw_ms):.2f}", f"{np.median(comp_ms):.2f}"],
        ["Std ms", f"{np.std(raw_ms):.2f}", f"{np.std(comp_ms):.2f}"],
        ["Min ms", f"{np.min(raw_ms):.2f}", f"{np.min(comp_ms):.2f}"],
        ["Max ms", f"{np.max(raw_ms):.2f}", f"{np.max(comp_ms):.2f}"],
        ["Samples", str(raw_ms.size), str(comp_ms.size)],
        ["Recommended", "", f"mean {rec['mean']:.1f} / median {rec['median']:.1f} ms"],
    ]
    table = axes[1, 2].table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    axes[1, 2].set_title("Summary")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return json_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate iPhone Record3D camera hardware latency using QR codes"
    )
    parser.add_argument("--device-index", type=int, default=0, help="Index into Record3D devices")
    parser.add_argument("--device-udid", type=str, default=None, help="Explicit Record3D device UDID")
    parser.add_argument("--qr-size", type=int, default=640, help="Displayed QR image size")
    parser.add_argument("--display-fps", type=float, default=60.0, help="QR display refresh loop FPS")
    parser.add_argument("--n-frames", type=int, default=500, help="Number of decoded samples to keep")
    parser.add_argument(
        "--display-latency-ms",
        type=float,
        default=DISPLAY_LATENCY_S * 1000.0,
        help="Display scanout/presentation latency to subtract",
    )
    parser.add_argument(
        "--shape",
        type=str,
        default=None,
        help="Optional processing shape, e.g. 480,640,3. Default uses Record3D frame size.",
    )
    parser.add_argument("--rotation-degree", type=int, default=0, choices=(0, 90, 180, 270))
    parser.add_argument("--no-flip", action="store_true", help="Disable TrueDepth horizontal flip")
    parser.add_argument(
        "--auto-stop",
        action="store_true",
        help="Stop automatically once n-frames valid detections are collected",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/latency_calibration",
        help="Directory for JSON and plot outputs",
    )
    args = parser.parse_args()

    cv2.setNumThreads(1)
    output_shape = parse_shape(args.shape)

    raw_latency_samples = deque(maxlen=args.n_frames)
    detection_samples = deque(maxlen=args.n_frames)
    qr_generation_times = deque(maxlen=args.n_frames)
    display_processing_times = deque(maxlen=args.n_frames)

    camera = IPhoneRecord3DCamera(
        device_index=args.device_index,
        device_udid=args.device_udid,
        flip_truedepth=not args.no_flip,
        rotation_degree=args.rotation_degree,
        output_shape=output_shape,
    )
    detector = cv2.QRCodeDetector()
    last_processed_frame_time: Optional[float] = None

    print("Starting iPhone latency calibration.")
    print("Show the QR window to the iPhone. Press 'c' to capture results, 'q' to quit.")
    camera.start()

    try:
        while True:
            loop_start = time.time()

            qr_gen_start = time.time()
            qr_timestamp_s = time.time()
            qr_img = make_timestamp_qr(qr_timestamp_s, args.qr_size)
            qr_gen_end = time.time()

            cv2.imshow("Timestamp QRCode", qr_img)
            display_done = time.time()

            qr_generation_times.append(qr_gen_end - qr_gen_start)
            display_processing_times.append(display_done - qr_gen_end)

            latest = camera.get_latest()
            latest_latency_s = float("nan")
            latest_corners = None
            if latest is not None:
                frame_receive_time_s, frame_rgb = latest
                if last_processed_frame_time != frame_receive_time_s:
                    last_processed_frame_time = frame_receive_time_s
                    code, latest_corners = decode_qr(detector, frame_rgb)
                    if code:
                        try:
                            embedded_time_s = float(code)
                            latest_latency_s = frame_receive_time_s - embedded_time_s
                            raw_latency_samples.append(latest_latency_s)
                            detection_samples.append(latest_latency_s)
                        except ValueError:
                            detection_samples.append(float("nan"))
                    else:
                        detection_samples.append(float("nan"))

                recent = list(detection_samples)[-30:]
                recent_rate = (
                    1.0 - float(np.mean(np.isnan(recent))) if recent else 0.0
                )
                vis = draw_camera_overlay(
                    frame_rgb,
                    latest_corners,
                    latest_latency_s,
                    recent_rate,
                    args.display_fps,
                )
                vis = cv2.resize(vis, (args.qr_size, args.qr_size), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("iPhone Camera Feed", vis)
            else:
                placeholder = np.zeros((args.qr_size, args.qr_size, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    "Waiting for iPhone frames...",
                    (40, args.qr_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("iPhone Camera Feed", placeholder)

            raw = finite_array(raw_latency_samples)
            avg_qr_gen = float(np.mean(qr_generation_times)) if qr_generation_times else QR_GENERATION_OVERHEAD_S
            avg_display_processing = (
                float(np.mean(display_processing_times)) if display_processing_times else 0.0
            )
            compensated = raw - avg_qr_gen - avg_display_processing - args.display_latency_ms / 1000.0
            det_rate = raw.size / max(len(detection_samples), 1)
            raw_mean = float(np.mean(raw)) if raw.size else float("nan")
            comp_mean = float(np.mean(compensated)) if compensated.size else float("nan")
            print(
                "Samples: {:4d}/{:<4d} | Det: {:5.1%} | Raw: {:7.2f} ms | "
                "Comp: {:7.2f} ms | QRGen: {:5.2f} ms".format(
                    raw.size,
                    args.n_frames,
                    det_rate,
                    raw_mean * 1000.0 if np.isfinite(raw_mean) else 0.0,
                    comp_mean * 1000.0 if np.isfinite(comp_mean) else 0.0,
                    avg_qr_gen * 1000.0,
                )
            )

            elapsed = time.time() - loop_start
            sleep_s = max(0.0, 1.0 / max(args.display_fps, 1e-6) - elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)

            key = cv2.pollKey()
            if key == ord("c"):
                break
            if key in (ord("q"), 27):
                print("Quitting without saving results.")
                return
            if args.auto_stop and raw.size >= args.n_frames:
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    raw = finite_array(raw_latency_samples)
    if raw.size == 0:
        print("No valid QR latency samples were collected.")
        return

    avg_qr_gen = float(np.mean(qr_generation_times)) if qr_generation_times else QR_GENERATION_OVERHEAD_S
    avg_display_processing = (
        float(np.mean(display_processing_times)) if display_processing_times else 0.0
    )
    compensated = raw - avg_qr_gen - avg_display_processing - args.display_latency_ms / 1000.0
    stats = build_stats(
        raw_latencies=raw,
        compensated_latencies=compensated,
        all_detection_values=detection_samples,
        qr_generation_times=qr_generation_times,
        display_processing_times=display_processing_times,
        args=args,
        camera_info=camera.connected_info(),
    )
    print_stats(stats)
    json_path, plot_path = save_results(
        stats=stats,
        raw_latencies=raw,
        compensated_latencies=compensated,
        output_dir=Path(args.output_dir).expanduser(),
    )
    print(f"Results saved to: {json_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
