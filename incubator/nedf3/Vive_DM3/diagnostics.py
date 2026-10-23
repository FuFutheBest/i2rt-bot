#!/usr/bin/env python3
"""
Active hardware diagnostics for the Vive_DM3 desktop GUI.

These helpers probe individual devices before the full collection stack starts.

Author: Yichen Xie
"""

from __future__ import annotations

import time
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import serial
import yaml

from incubator.nedf3.Vive_DM3.process_manager import _compose_hydra_config, load_config_defaults
from devices.rotary_encoder import (
    _MODBUS_FRAME_LEN,
    _MODBUS_RESPONSE_PREFIX,
    _MODBUS_SYNC_READ_TIMEOUT_S,
    _validate_modbus_angle_frame,
)
from utils.gripper_angle_calibration import (
    GripperAngleCalibration,
    load_today_angle_calibration,
    process_gripper_angle_with_date_check,
)
from utils.logger_config import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_PROBE_TIMEOUT_SEC = 8.0


@dataclass
class CameraDiagnosticResult:
    """Result of probing one OpenCV camera index."""

    label: str
    capture_id: str
    ok: bool
    message: str
    frame_rgb: Optional[Any]
    image_png_b64: Optional[str] = None


@dataclass
class EncoderDiagnosticResult:
    """Result of sampling one rotary encoder serial port."""

    side: str
    port: str
    baudrate: int
    target_fps: float
    ok: bool
    message: str
    hz: float
    samples: List[Tuple[float, float]]
    error_count: int


def load_diagnostic_config(config_name: str) -> Dict[str, Any]:
    """Load active-diagnostic settings from a Hydra config."""
    defaults = load_config_defaults(config_name)
    result: Dict[str, Any] = {
        "camera_indices": dict(defaults.get("camera_indices") or {}),
        "encoder_ports": dict(defaults.get("encoder_ports") or {}),
        "encoder_baudrates": {"left": 1000000, "right": 1000000},
        "encoder_fps": {"left": 30.0, "right": 30.0},
        "tracker_roles": [],
    }
    try:
        config = _compose_hydra_config(config_name, [])
    except Exception as exc:
        logger.warning(f"Failed to compose diagnostic config {config_name}: {exc}")
        _load_diagnostic_config_from_yaml(config_name, result)
        return result

    for device in config.get("devices", []):
        if not isinstance(device, dict):
            continue
        device_class = str(device.get("class") or "")
        if device_class == "RotaryEncoderDevice":
            side = "left" if int(device.get("device_class_id", 0) or 0) == 0 else "right"
            baudrate = device.get("baudrate")
            state_feedback = device.get("state_feedback")
            state_feedback = state_feedback if isinstance(state_feedback, dict) else {}
            fps = state_feedback.get("fps", device.get("fps"))
            try:
                result["encoder_baudrates"][side] = int(baudrate)
            except (TypeError, ValueError):
                pass
            try:
                result["encoder_fps"][side] = float(fps)
            except (TypeError, ValueError):
                pass
        elif device_class == "ViveTrackerDevice":
            roles = device.get("tracker_roles")
            if isinstance(roles, list):
                result["tracker_roles"] = [str(role) for role in roles]
    if not result.get("tracker_roles"):
        _load_diagnostic_config_from_yaml(config_name, result)
    return result


def _load_diagnostic_config_from_yaml(config_name: str, result: Dict[str, Any]) -> None:
    """Best-effort fallback that avoids requiring Hydra in lightweight probes."""
    config_path = PROJECT_ROOT / "configs" / f"{config_name}.yaml"
    if not config_path.is_file():
        return
    try:
        with config_path.open(encoding="utf-8") as handle:
            top_config = yaml.safe_load(handle) or {}
    except Exception as exc:
        logger.warning(f"Failed to read diagnostic config yaml {config_path}: {exc}")
        return

    device_config_paths: List[Path] = []
    defaults = top_config.get("defaults")
    if isinstance(defaults, list):
        for item in defaults:
            if not isinstance(item, dict):
                continue
            for group_name, preset_name in item.items():
                group = str(group_name).lstrip("/").split("@", maxsplit=1)[0]
                if group in {"devices", "devices/infer"} and isinstance(
                    preset_name, str
                ):
                    device_config_paths.append(
                        PROJECT_ROOT / "configs" / group / f"{preset_name}.yaml"
                    )
    if isinstance(top_config.get("devices"), list):
        _merge_device_diagnostic_config(top_config.get("devices"), result)

    for devices_path in device_config_paths:
        if not devices_path.is_file():
            continue
        try:
            with devices_path.open(encoding="utf-8") as handle:
                devices_config = yaml.safe_load(handle) or {}
        except Exception as exc:
            logger.warning(f"Failed to read device diagnostic yaml {devices_path}: {exc}")
            continue
        devices = devices_config.get("devices")
        if isinstance(devices, list):
            _merge_device_diagnostic_config(devices, result)


def _merge_device_diagnostic_config(devices: Any, result: Dict[str, Any]) -> None:
    for device in devices:
        if not isinstance(device, dict):
            continue
        device_class = str(device.get("class") or "")
        if device_class == "RotaryEncoderDevice":
            side = "left" if int(device.get("device_class_id", 0) or 0) == 0 else "right"
            baudrate = device.get("baudrate")
            state_feedback = device.get("state_feedback")
            state_feedback = state_feedback if isinstance(state_feedback, dict) else {}
            fps = state_feedback.get("fps", device.get("fps"))
            try:
                result["encoder_baudrates"][side] = int(baudrate)
            except (TypeError, ValueError):
                pass
            try:
                result["encoder_fps"][side] = float(fps)
            except (TypeError, ValueError):
                pass
        elif device_class == "ViveTrackerDevice":
            roles = device.get("tracker_roles")
            if isinstance(roles, list):
                result["tracker_roles"] = [str(role) for role in roles]


def run_camera_diagnostics(camera_indices: Dict[str, str]) -> Dict[str, CameraDiagnosticResult]:
    """Open each configured camera index and capture one frame."""
    labels = {
        "right_capture_id": "right_wrist",
        "left_capture_id": "left_wrist",
        "ego_capture_id": "head",
    }
    results: Dict[str, CameraDiagnosticResult] = {}
    for key, label in labels.items():
        results[key] = _probe_camera_index(
            label,
            str(camera_indices.get(key) or "").strip(),
        )
    return results


def _probe_camera_index(label: str, capture_id: str) -> CameraDiagnosticResult:
    if not capture_id:
        return CameraDiagnosticResult(
            label=label,
            capture_id="",
            ok=False,
            message="capture_id is empty",
            frame_rgb=None,
        )
    try:
        index = int(capture_id)
    except ValueError:
        return CameraDiagnosticResult(
            label=label,
            capture_id=capture_id,
            ok=False,
            message="capture_id is not an integer",
            frame_rgb=None,
        )

    script_path = PROJECT_ROOT / "scripts" / "camera_frame_probe.py"
    process: Optional[subprocess.Popen[str]] = None
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(script_path),
                "--index",
                str(index),
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            stdout, stderr = process.communicate(timeout=CAMERA_PROBE_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except Exception:
                pass
            return CameraDiagnosticResult(
                label=label,
                capture_id=capture_id,
                ok=False,
                message=f"camera probe timed out after {CAMERA_PROBE_TIMEOUT_SEC:.0f}s",
                frame_rgb=None,
            )
    except subprocess.TimeoutExpired:
        return CameraDiagnosticResult(
            label=label,
            capture_id=capture_id,
            ok=False,
            message=f"camera probe timed out after {CAMERA_PROBE_TIMEOUT_SEC:.0f}s",
            frame_rgb=None,
        )
    except Exception as exc:
        return CameraDiagnosticResult(
            label=label,
            capture_id=capture_id,
            ok=False,
            message=str(exc),
            frame_rgb=None,
        )

    returncode = process.returncode if process is not None else -1
    raw_output = (stdout or "").strip().splitlines()
    payload: Dict[str, Any]
    try:
        payload = json.loads(raw_output[-1] if raw_output else "{}")
    except Exception:
        payload = {}
    if returncode != 0 or not payload.get("ok"):
        message = str(payload.get("message") or stderr or f"probe exit={returncode}").strip()
        return CameraDiagnosticResult(
            label=label,
            capture_id=capture_id,
            ok=False,
            message=message,
            frame_rgb=None,
        )

    image_png_b64 = str(payload.get("image_png_b64") or "")
    if not image_png_b64:
        return CameraDiagnosticResult(
            label=label,
            capture_id=capture_id,
            ok=False,
            message="probe returned no image",
            frame_rgb=None,
        )
    dark_preview = bool(payload.get("dark_preview"))
    return CameraDiagnosticResult(
        label=label,
        capture_id=capture_id,
        ok=not dark_preview,
        message=str(payload.get("message") or "OK"),
        frame_rgb=None,
        image_png_b64=image_png_b64,
    )


def run_encoder_diagnostics(
    encoder_ports: Dict[str, str],
    encoder_baudrates: Dict[str, int],
    encoder_fps: Optional[Dict[str, float]] = None,
    duration_sec: float = 5.0,
) -> Dict[str, EncoderDiagnosticResult]:
    """Sample configured rotary encoders and estimate read frequency."""
    target_fps = encoder_fps or {}
    specs = {
        "left": (
            str(encoder_ports.get("left_port") or "").strip(),
            int(encoder_baudrates.get("left", 1000000)),
            float(target_fps.get("left", 30.0)),
        ),
        "right": (
            str(encoder_ports.get("right_port") or "").strip(),
            int(encoder_baudrates.get("right", 1000000)),
            float(target_fps.get("right", 30.0)),
        ),
    }
    calibrations: Dict[str, Optional[GripperAngleCalibration]] = {}
    calibration_errors: Dict[str, str] = {}
    today = date.today().isoformat()
    for side in ("left", "right"):
        try:
            calibrations[side] = load_today_angle_calibration(
                str(PROJECT_ROOT),
                side,
                expected_date=today,
            )
            calibration_errors[side] = ""
        except Exception as exc:
            calibrations[side] = None
            calibration_errors[side] = str(exc)

    results: Dict[str, EncoderDiagnosticResult] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            side: executor.submit(
                _sample_encoder_port,
                side,
                port,
                baudrate,
                fps,
                duration_sec,
                calibrations.get(side),
                calibration_errors.get(side, ""),
                today,
            )
            for side, (port, baudrate, fps) in specs.items()
        }
        for side, future in futures.items():
            results[side] = future.result()
    return results


def _sample_encoder_port(
    side: str,
    port: str,
    baudrate: int,
    target_fps: float,
    duration_sec: float,
    calibration: Optional[GripperAngleCalibration],
    calibration_error: str,
    calibration_date: str,
) -> EncoderDiagnosticResult:
    target_fps = max(1.0, float(target_fps))
    if not port:
        return EncoderDiagnosticResult(
            side=side,
            port="",
            baudrate=baudrate,
            target_fps=target_fps,
            ok=False,
            message="serial port is empty",
            hz=0.0,
            samples=[],
            error_count=0,
        )
    if calibration is None:
        return EncoderDiagnosticResult(
            side=side,
            port=port,
            baudrate=baudrate,
            target_fps=target_fps,
            ok=False,
            message=f"calibration unavailable: {calibration_error}",
            hz=0.0,
            samples=[],
            error_count=1,
        )

    samples: List[Tuple[float, float]] = []
    error_count = 0
    start = time.perf_counter()
    try:
        conn = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.02,
        )
    except Exception as exc:
        return EncoderDiagnosticResult(
            side=side,
            port=port,
            baudrate=baudrate,
            target_fps=target_fps,
            ok=False,
            message=f"failed to open serial port: {exc}",
            hz=0.0,
            samples=[],
            error_count=1,
        )

    try:
        read_interval = 1.0 / target_fps
        next_read_at = start
        while time.perf_counter() - start < duration_sec:
            raw_angle = _read_encoder_angle(conn)
            elapsed = time.perf_counter() - start
            if raw_angle is None:
                error_count += 1
            else:
                try:
                    angle = process_gripper_angle_with_date_check(
                        raw_angle,
                        calibration,
                        calibration_date,
                    )
                except Exception:
                    error_count += 1
                else:
                    samples.append((elapsed, angle))
            next_read_at += read_interval
            sleep_sec = next_read_at - time.perf_counter()
            if sleep_sec > 0:
                time.sleep(sleep_sec)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    hz = _estimate_hz(samples)
    ok = bool(samples)
    message = (
        f"OK calibrated samples={len(samples)} hz={hz:.1f} target={target_fps:.1f}"
        if ok
        else "no valid encoder samples"
    )
    return EncoderDiagnosticResult(
        side=side,
        port=port,
        baudrate=baudrate,
        target_fps=target_fps,
        ok=ok,
        message=message,
        hz=hz,
        samples=samples,
        error_count=error_count,
    )


def _read_encoder_angle(conn: serial.Serial) -> Optional[float]:
    try:
        conn.reset_input_buffer()
        conn.write(bytes.fromhex("01 03 00 41 00 01 d4 1e"))
        rx_buffer = bytearray()
        deadline = time.monotonic() + _MODBUS_SYNC_READ_TIMEOUT_S
        while time.monotonic() < deadline:
            pending = conn.in_waiting
            chunk = conn.read(pending if pending > 0 else 1)
            if chunk:
                rx_buffer.extend(chunk)
            while True:
                sync_idx = rx_buffer.find(_MODBUS_RESPONSE_PREFIX)
                if sync_idx < 0 or len(rx_buffer) < sync_idx + _MODBUS_FRAME_LEN:
                    break
                frame = bytes(rx_buffer[sync_idx:sync_idx + _MODBUS_FRAME_LEN])
                if _validate_modbus_angle_frame(frame):
                    return _calculate_encoder_angle(frame)
                del rx_buffer[sync_idx]
            if not chunk:
                time.sleep(0.001)
    except Exception:
        return None
    return None


def _calculate_encoder_angle(frame: bytes) -> Optional[float]:
    if len(frame) < 7:
        return None
    angle_raw = (frame[3] << 8) | frame[4]
    mask = 0b111111111111
    sign_bit = 0b100000000000
    angle_raw = angle_raw & mask
    if angle_raw & sign_bit:
        angle_raw = -((~angle_raw & mask) + 1)
    return 360.0 * angle_raw / 4096.0


def _estimate_hz(samples: List[Tuple[float, float]]) -> float:
    if len(samples) < 2:
        return 0.0
    elapsed = max(1e-6, samples[-1][0] - samples[0][0])
    return (len(samples) - 1) / elapsed
