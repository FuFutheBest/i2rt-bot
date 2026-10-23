"""Gripper backends for configurable I2RT teleoperation."""

from __future__ import annotations

import fcntl
import json
import logging
import math
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from utils.lk_gripper_presets import get_lk_gripper_preset

from .config import REPO_ROOT, GripperConfig

LOGGER = logging.getLogger(__name__)
LKTECH_RECONNECT_INTERVAL_S = 2.0
LKTECH_SHUTDOWN_TIMEOUT_S = 5.0


class GripperBackend(Protocol):
    def start(self) -> None: ...

    def read_normalized(self) -> float: ...

    def command_normalized(self, value: float) -> bool: ...

    def close(self) -> None: ...


class NullGripper:
    """A gripper backend that exposes a stable placeholder state."""

    def start(self) -> None:
        return None

    def read_normalized(self) -> float:
        return 0.0

    def command_normalized(self, value: float) -> bool:
        _finite_normalized(value, "gripper command")
        return False

    def close(self) -> None:
        return None


def _finite_normalized(value: float, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return float(np.clip(result, 0.0, 1.0))


def _calibration_number(
    calibration: dict[str, Any], key: str, fallback: float | int | None = None
) -> float | int | None:
    value = calibration.get(key, fallback)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"LKTech calibration field {key} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"LKTech calibration field {key} must be finite")
    return numeric


class LKTechGripper:
    """Nonblocking teleoperation adapter around the bundled LKTech wrapper."""

    def __init__(
        self,
        config: GripperConfig,
        *,
        label: str,
        driver: Any | None = None,
        reconnect_interval_s: float = LKTECH_RECONNECT_INTERVAL_S,
    ) -> None:
        if config.backend != "lktech":
            raise ValueError("LKTechGripper requires backend='lktech'")
        if reconnect_interval_s <= 0.0:
            raise ValueError("LKTech reconnect interval must be positive")
        self.config = config
        self.label = label
        self._driver = driver
        self._reconnect_interval_s = float(reconnect_interval_s)
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._started = False
        self._connected = False
        self._health = "stopped"
        self._fault_active = False
        self._last_error: str | None = None
        self._startup_value = (
            None
            if config.startup_width_ratio is None
            else _finite_normalized(
                config.startup_width_ratio, "startup gripper command"
            )
        )
        self._desired_value = self._startup_value
        self._desired_generation = 0
        self._applied_generation = 0
        self._command_floor_generation = 0
        self._startup_pending = self._startup_value is not None
        self._last_feedback_value = (
            self._desired_value if self._desired_value is not None else 0.0
        )
        self._lock_handle = None
        self._min_width_m = float(config.min_width_m)
        self._max_width_m: float | None = config.max_width_m

    @staticmethod
    def _load_driver_classes():
        sdk_root = REPO_ROOT / "third_party" / "lktech_motor_sdk"
        if str(sdk_root) not in sys.path:
            sys.path.insert(0, str(sdk_root))
        from third_party.lktech_motor_sdk.example.lktech_gripper import (
            Gripper,
            LKGripperConfig,
        )

        return LKGripperConfig, Gripper

    def _load_calibration(self) -> dict[str, Any]:
        path = self.config.calibration_file
        if path is None:
            raise ValueError(f"{self.label} LKTech calibration file is required")
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"{self.label} LKTech calibration must be a JSON object")
        return document

    def _build_driver(self):
        calibration = self._load_calibration()
        preset = get_lk_gripper_preset(
            str(calibration.get("gripper_type", self.config.gripper_type))
        )
        config_cls, gripper_cls = self._load_driver_classes()
        runtime_config = config_cls(
            port=str(self.config.port),
            motor_id=self.config.motor_id,
            baudrate=self.config.baudrate,
            speed_dps=self.config.speed_dps,
            auto_calibrate=False,
            min_torque_iq=int(
                _calibration_number(
                    calibration, "min_torque_iq", preset["min_torque_iq"]
                )
            ),
            max_torque_iq=int(
                _calibration_number(
                    calibration, "max_torque_iq", preset["max_torque_iq"]
                )
            ),
            min_encoder=_calibration_number(calibration, "min_encoder"),
            max_encoder=_calibration_number(calibration, "max_encoder"),
            min_angle_deg=_calibration_number(calibration, "min_angle_deg"),
            max_angle_deg=_calibration_number(calibration, "max_angle_deg"),
            arm_length_m=float(
                _calibration_number(
                    calibration, "arm_length_m", preset["arm_length_m"]
                )
            ),
            apex_angle_max_deg=float(
                _calibration_number(
                    calibration,
                    "apex_angle_max_deg",
                    preset["apex_angle_max_deg"],
                )
            ),
            gear_ratio=float(_calibration_number(calibration, "gear_ratio", 1.0)),
            label=self.label,
        )
        return gripper_cls(
            robot=None,
            config=runtime_config,
            min_vel=0.0,
            max_vel=max(1.0, self.config.command_velocity),
        )

    def _acquire_port_lock(self) -> None:
        safe_port = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(self.config.port))
        lock_path = Path("/tmp") / f"superinference_lktech_{safe_port}.lock"
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(
                "LKTech serial port is already owned by another process: "
                f"{self.config.port}"
            ) from exc
        self._lock_handle = handle

    def _release_port_lock(self) -> None:
        if self._lock_handle is None:
            return
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._lock_handle.close()
            self._lock_handle = None

    def start(self) -> None:
        if self._started:
            return
        self._acquire_port_lock()
        try:
            if self._driver is None:
                self._driver = self._build_driver()
            sdk_max_width = float(self._driver.max_width_m)
            if not math.isfinite(sdk_max_width) or sdk_max_width <= 0.0:
                raise ValueError(f"{self.label} LKTech SDK reported invalid max width")
            if self._max_width_m is None:
                self._max_width_m = sdk_max_width
            else:
                self._max_width_m = min(float(self._max_width_m), sdk_max_width)
            if self._max_width_m <= self._min_width_m:
                raise ValueError(
                    f"{self.label} LKTech maximum width must exceed minimum width"
                )
            with self._condition:
                self._stop_requested = False
                self._started = True
                self._health = "starting"
                self._thread = threading.Thread(
                    target=self._run_io,
                    name=f"{self.label}-io",
                    daemon=True,
                )
                self._thread.start()
        except Exception:
            with self._condition:
                self._started = False
                self._health = "stopped"
                self._thread = None
            self._release_port_lock()
            raise

    def _require_started(self) -> None:
        if not self._started or self._driver is None or self._max_width_m is None:
            raise RuntimeError(f"{self.label} LKTech gripper is not started")

    @property
    def health(self) -> str:
        with self._condition:
            return self._health

    @property
    def last_error(self) -> str | None:
        with self._condition:
            return self._last_error

    def _operator_to_physical_ratio(self, value: float) -> float:
        operator_value = _finite_normalized(value, "gripper command")
        if self.config.invert:
            operator_value = 1.0 - operator_value
        return float(
            np.clip(
                operator_value * self.config.scale_factor
                - self.config.command_offset,
                0.0,
                1.0,
            )
        )

    def _physical_to_operator_ratio(self, value: float) -> float:
        physical_value = _finite_normalized(value, "gripper feedback")
        operator_value = (
            physical_value + self.config.command_offset
        ) / self.config.scale_factor
        operator_value = float(np.clip(operator_value, 0.0, 1.0))
        return 1.0 - operator_value if self.config.invert else operator_value

    def command_normalized(self, value: float) -> bool:
        operator_value = _finite_normalized(value, "gripper command")
        with self._condition:
            self._require_started()
            if (
                self._desired_value is not None
                and abs(operator_value - self._desired_value) < self.config.deadband
            ):
                return False
            self._desired_value = operator_value
            self._desired_generation += 1
            self._condition.notify_all()
            return True

    def read_normalized(self) -> float:
        with self._condition:
            self._require_started()
            return self._last_feedback_value

    def _command_driver(self, operator_value: float) -> None:
        ratio = self._operator_to_physical_ratio(operator_value)
        width = self._min_width_m + ratio * (
            float(self._max_width_m) - self._min_width_m
        )
        if not self._driver.Move(width, self.config.command_velocity, None):
            raise RuntimeError(f"{self.label} LKTech gripper rejected command")

    def _read_driver_feedback(self) -> float:
        width = float(self._driver.states().width)
        width = float(np.clip(width, self._min_width_m, float(self._max_width_m)))
        physical_ratio = (width - self._min_width_m) / (
            float(self._max_width_m) - self._min_width_m
        )
        return self._physical_to_operator_ratio(physical_ratio)

    def _wait(self, timeout: float) -> None:
        with self._condition:
            if not self._stop_requested:
                self._condition.wait(timeout=max(0.0, timeout))

    def _is_stopping(self) -> bool:
        with self._condition:
            return self._stop_requested

    def _safe_disable(self) -> None:
        try:
            self._driver.Disable()
        except Exception as exc:
            LOGGER.debug("%s disable failed during recovery: %s", self.label, exc)

    def _record_fault(self, operation: str, exc: Exception) -> None:
        with self._condition:
            first_failure = not self._fault_active
            self._fault_active = True
            self._connected = False
            self._health = "recovering"
            self._last_error = f"{operation}: {exc}"
            self._startup_pending = False
        log = LOGGER.warning if first_failure else LOGGER.debug
        log(
            "%s %s failed; arm teleoperation remains active and gripper "
            "reconnect will be attempted: %s",
            self.label,
            operation,
            exc,
        )

    def _record_recovery(self) -> None:
        with self._condition:
            if not self._fault_active:
                return
            self._fault_active = False
            self._health = "healthy"
            self._last_error = None
        LOGGER.info(
            "%s communication recovered; waiting for new leader movement",
            self.label,
        )

    def _run_io(self) -> None:
        command_interval = 1.0 / self.config.command_hz
        feedback_interval = 1.0 / self.config.feedback_hz
        connected = False
        next_connect_at = 0.0
        next_command_at = 0.0
        next_feedback_at = 0.0

        try:
            while not self._is_stopping():
                now = time.monotonic()
                if not connected:
                    if now < next_connect_at:
                        self._wait(min(next_connect_at - now, 0.1))
                        continue
                    try:
                        self._driver.Enable(self.label)
                    except Exception as exc:
                        self._record_fault("connection", exc)
                        self._safe_disable()
                        next_connect_at = (
                            time.monotonic() + self._reconnect_interval_s
                        )
                        continue

                    with self._condition:
                        recovering = self._fault_active
                        if recovering:
                            self._command_floor_generation = self._desired_generation
                            self._applied_generation = max(
                                self._applied_generation,
                                self._command_floor_generation,
                            )
                            self._startup_pending = False
                            self._health = "recovering"
                        else:
                            self._health = "healthy"
                        self._connected = True
                    connected = True
                    next_command_at = time.monotonic()
                    next_feedback_at = next_command_at
                    LOGGER.info("%s serial worker connected", self.label)

                now = time.monotonic()
                with self._condition:
                    startup_pending = self._startup_pending
                    desired_value = self._desired_value
                    desired_generation = self._desired_generation
                    command_pending = startup_pending or (
                        desired_value is not None
                        and desired_generation > self._command_floor_generation
                        and desired_generation > self._applied_generation
                    )

                if command_pending and now >= next_command_at:
                    command_value = (
                        self._startup_value if startup_pending else desired_value
                    )
                    try:
                        self._command_driver(float(command_value))
                    except Exception as exc:
                        self._record_fault("command", exc)
                        self._safe_disable()
                        connected = False
                        next_connect_at = (
                            time.monotonic() + self._reconnect_interval_s
                        )
                        continue
                    with self._condition:
                        if startup_pending:
                            self._startup_pending = False
                        else:
                            self._applied_generation = max(
                                self._applied_generation, desired_generation
                            )
                    next_command_at = time.monotonic() + command_interval

                now = time.monotonic()
                if now >= next_feedback_at:
                    try:
                        feedback = self._read_driver_feedback()
                    except Exception as exc:
                        self._record_fault("feedback", exc)
                        self._safe_disable()
                        connected = False
                        next_connect_at = (
                            time.monotonic() + self._reconnect_interval_s
                        )
                        continue
                    with self._condition:
                        self._last_feedback_value = feedback
                    self._record_recovery()
                    next_feedback_at = time.monotonic() + feedback_interval

                now = time.monotonic()
                with self._condition:
                    command_pending = self._startup_pending or (
                        self._desired_value is not None
                        and self._desired_generation
                        > self._command_floor_generation
                        and self._desired_generation > self._applied_generation
                    )
                deadlines = [next_feedback_at]
                if command_pending:
                    deadlines.append(next_command_at)
                timeout = max(0.001, min(deadlines) - now)
                self._wait(min(timeout, 0.1))
        finally:
            self._safe_disable()
            with self._condition:
                self._connected = False
                self._health = "stopped"
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if not self._started and self._thread is None:
                self._release_port_lock()
                return
            self._stop_requested = True
            self._started = False
            self._health = "stopping"
            thread = self._thread
            self._condition.notify_all()

        if thread is not None:
            thread.join(timeout=LKTECH_SHUTDOWN_TIMEOUT_S)
        if thread is not None and thread.is_alive():
            LOGGER.error("%s serial worker did not stop cleanly", self.label)
            return

        with self._condition:
            self._thread = None
            self._release_port_lock()
