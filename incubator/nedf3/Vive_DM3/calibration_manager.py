#!/usr/bin/env python3
"""
Gripper angle calibration for the Vive_DM3 desktop GUI.

Mirrors the Windows calibration flow in ``scripts/data_collection_gui.py``.

Author: Yichen Xie, Zixi Ying
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from devices.rotary_encoder import RotaryEncoderDevice
from utils.gripper_angle_calibration import (
    DEFAULT_READ_RATE_HZ,
    build_calibration_from_recording,
    build_encoder_runtime_config,
    get_angle_calibration_path,
    save_angle_calibration,
)
from utils.time_control import precise_loop_timing

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CalibrationManager:
    """Thread-safe gripper encoder calibration session."""

    def __init__(self, device_is_running: Callable[[], bool]) -> None:
        self._device_is_running = device_is_running
        self._lock = threading.RLock()
        self._active = False
        self._recording = False
        self._side = "left"
        self._encoder: Optional[RotaryEncoderDevice] = None
        self._thread: Optional[threading.Thread] = None
        self._raw: List[float] = []
        self._timestamps: List[float] = []
        self._message = "选择左右手后点击「开始标定」；需先停止设备启动以独占串口"
        self._saving = False
        self._save_complete_callback: Optional[Callable[[bool, str], None]] = None

    def set_save_complete_callback(
        self, callback: Optional[Callable[[bool, str], None]]
    ) -> None:
        """Register a callback invoked after calibration save finishes (desktop GUI)."""
        self._save_complete_callback = callback

    def status_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active": self._active,
                "recording": self._recording,
                "saving": self._saving,
                "side": self._side,
                "message": self._message,
                "sample_count": len(self._raw) if self._recording else 0,
            }

    def start(
        self,
        *,
        config_name: str,
        side: str,
        encoder_ports: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        side = side.strip().lower()
        if side not in ("left", "right"):
            raise ValueError("请选择左或右夹爪。")
        if self._active:
            raise RuntimeError("标定已在进行中。")
        if self._device_is_running():
            raise RuntimeError("标定需要独占串口，请先停止设备启动。")
        if not config_name.strip():
            raise ValueError("请选择配置文件。")

        runtime_cfg = build_encoder_runtime_config(
            str(PROJECT_ROOT),
            config_name.strip(),
            side,
        )
        ports = encoder_ports or {}
        port_key = "left_port" if side == "left" else "right_port"
        selected_port = str(ports.get(port_key) or "").strip()
        if selected_port:
            runtime_cfg["port"] = selected_port

        encoder = RotaryEncoderDevice(
            device_class_id=0 if side == "left" else 1,
            state_feedback={
                "fps": float(runtime_cfg["read_rate_hz"]),
                "buffer_size": 100,
                "hardware_latency_ms": 5.0,
                "format": {
                    "shape": [1],
                    "dtype": "float64",
                    "type": ["gripper"],
                },
            },
            port=str(runtime_cfg["port"]),
            baudrate=int(runtime_cfg["baudrate"]),
            calibration_file=str(runtime_cfg["calibration_file"]),
        )
        if encoder.serial_conn is None:
            raise RuntimeError(
                f"无法打开 {runtime_cfg['port']}。"
                "请检查接线、波特率，并确认无其他程序占用该 COM 口。"
            )

        with self._lock:
            self._encoder = encoder
            self._side = side
            self._raw = []
            self._timestamps = []
            self._recording = False
            self._active = True
            self._message = (
                "标定就绪：闭合→张开移动夹爪，按回车或点击按钮开始录制，"
                "再按一次停止并保存"
            )

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self.status_dict()

    def stop(self, *, message: Optional[str] = None) -> Dict[str, Any]:
        self._active = False
        encoder = self._encoder
        self._encoder = None
        if encoder is not None and encoder.serial_conn is not None:
            try:
                if encoder.serial_conn.is_open:
                    encoder.serial_conn.close()
            except Exception:
                pass

        with self._lock:
            self._recording = False
            self._saving = False
            self._raw = []
            self._timestamps = []
            self._message = message or (
                "选择左右手后点击「开始标定」；需先停止设备启动以独占串口"
            )

        return self.status_dict()

    def toggle_recording(self) -> Dict[str, Any]:
        if not self._active:
            raise RuntimeError("标定未开始。")
        if self._saving:
            raise RuntimeError("标定数据正在保存，请稍候。")

        with self._lock:
            if not self._recording:
                self._recording = True
                self._raw = []
                self._timestamps = []
                self._message = (
                    "录制中：请从最小张开度移动到最大张开度，"
                    "再按回车或点击按钮停止保存"
                )
                return {
                    "active": self._active,
                    "recording": self._recording,
                    "saving": self._saving,
                    "side": self._side,
                    "message": self._message,
                    "sample_count": 0,
                }

            self._recording = False
            raw_copy = list(self._raw)
            ts_copy = list(self._timestamps)
            side = self._side
            self._saving = True
            self._message = "正在保存标定数据..."

        threading.Thread(
            target=self._finalize_recording,
            args=(raw_copy, ts_copy, side),
            daemon=True,
        ).start()
        return self.status_dict()

    def _finalize_recording(
        self,
        raw_copy: List[float],
        ts_copy: List[float],
        side: str,
    ) -> None:
        success = False
        final_message = ""
        try:
            calibration = build_calibration_from_recording(raw_copy, ts_copy)
            output_path = get_angle_calibration_path(str(PROJECT_ROOT), side)
            save_angle_calibration(calibration, output_path)
            with self._lock:
                self._message = (
                    f"已保存 {side} 夹爪标定：{output_path} "
                    f"(span={calibration.span_deg:.2f}°)"
                )
            success = True
        except ValueError as exc:
            with self._lock:
                self._message = f"标定失败：{exc}"
        finally:
            with self._lock:
                final_message = self._message
                self._saving = False
                callback = self._save_complete_callback
            self.stop(message=final_message)
            if callback is not None:
                callback(success, final_message)

    def _reader_loop(self) -> None:
        wait_for_next = precise_loop_timing(1.0 / DEFAULT_READ_RATE_HZ)
        while self._active and self._encoder is not None:
            angle = self._encoder._get_angle_data()
            if angle is not None:
                with self._lock:
                    if self._recording:
                        self._raw.append(float(angle))
                        self._timestamps.append(time.time())
            wait_for_next()
