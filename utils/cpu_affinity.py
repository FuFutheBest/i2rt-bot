#!/usr/bin/env python3
"""
CPU affinity helpers for DeviceStarter worker processes.

Layouts can be overridden via ``runtime.cpu_affinity`` in Hydra config; when unset,
presets matching the historical hard-coded 12-core / 8-core maps are used.

Author: Zixi Ying
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import psutil

logger = logging.getLogger(__name__)

RoleName = str

DEFAULT_MIN_CPUS = 8

# Historical layouts (same as former device_starter.py hard-coding).
_PRESET_12CPU: Dict[str, Any] = {
    "device_starter_main": [0, 1],
    "device_manager": [0, 1],
    "policy_runner": [2, 3, 4],
    "visualizer": [7],
    "device_default": [0, 1, 2, 3, 4],
    "devices": {
        "RizonRobot": [8, 9, 10, 11],
        "HikCameraDevice": [5, 6],
    },
}

_PRESET_8CPU: Dict[str, Any] = {
    "device_starter_main": [0],
    "device_manager": [0],
    "policy_runner": [1],
    "visualizer": [2],
    "device_default": [0, 1],
    "devices": {
        "RizonRobot": [4, 5, 6, 7],
        "HikCameraDevice": [2, 3],
    },
}


def _as_int_list(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None
    out: List[int] = []
    for item in value:
        out.append(int(item))
    return out


def _valid_cpu_affinity(cores: Sequence[int]) -> Optional[List[int]]:
    n = psutil.cpu_count()
    if not n:
        return None
    masked = [int(c) for c in cores if 0 <= int(c) < n]
    return masked if masked else None


def pin_process_or_warn(cores: Sequence[int], *, role: str) -> None:
    """Apply CPU affinity to the current process; log and continue on failure."""
    allowed = _valid_cpu_affinity(cores)
    if not allowed:
        logger.warning(f"CPU affinity skipped for {role}: no valid cores in {list(cores)}")
        return
    try:
        psutil.Process().cpu_affinity(allowed)
        logger.info(f"CPU affinity ({role}): PID {os.getpid()} -> cores {allowed}")
    except OSError as exc:
        logger.warning(f"CPU affinity failed for {role}: {exc}")


def _pick_preset_layout(logical_cpus: int) -> Optional[Dict[str, Any]]:
    if logical_cpus >= 12:
        return dict(_PRESET_12CPU)
    if logical_cpus >= DEFAULT_MIN_CPUS:
        return dict(_PRESET_8CPU)
    return None


@dataclass
class CpuAffinitySettings:
    """Resolved CPU affinity policy for DeviceStarter subprocesses."""

    enabled: bool = True
    min_cpus: int = DEFAULT_MIN_CPUS
    profile: str = "auto"
    layout: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_runtime(cls, runtime_cfg: Optional[Mapping[str, Any]]) -> "CpuAffinitySettings":
        raw = dict(runtime_cfg or {}).get("cpu_affinity") or {}
        if not isinstance(raw, dict):
            raw = {}

        enabled = bool(raw.get("enabled", True))
        min_cpus = int(raw.get("min_cpus", DEFAULT_MIN_CPUS))
        profile = str(raw.get("profile", "auto")).strip().lower()

        layout: Dict[str, Any] = {}
        if profile == "custom":
            custom = raw.get("custom") or raw.get("roles") or {}
            if isinstance(custom, dict):
                layout = dict(custom)
        elif profile in ("12cpu", "12"):
            layout = dict(_PRESET_12CPU)
        elif profile in ("8cpu", "8"):
            layout = dict(_PRESET_8CPU)
        elif profile == "auto":
            layout = {}
        else:
            logger.warning(
                "Unknown cpu_affinity.profile={!r}; falling back to auto presets",
                profile,
            )
            profile = "auto"

        return cls(enabled=enabled, min_cpus=min_cpus, profile=profile, layout=layout)

    def _resolved_layout(self) -> Optional[Dict[str, Any]]:
        n = psutil.cpu_count() or 1
        if not self.enabled:
            return None
        if n < self.min_cpus:
            return None
        if self.profile == "auto":
            return _pick_preset_layout(n)
        if not self.layout:
            if self.profile == "custom":
                raise ValueError(
                    "cpu_affinity.profile='custom' requires a non-empty layout "
                    "(set runtime.cpu_affinity.custom or runtime.cpu_affinity.roles)"
                )
            logger.warning(
                "CPU affinity profile={!r} has no layout; affinity disabled for this process",
                self.profile,
            )
            return None
        return self.layout

    def cores_for_role(
        self,
        role: RoleName,
        *,
        device_class: Optional[str] = None,
    ) -> Optional[List[int]]:
        layout = self._resolved_layout()
        if layout is None:
            return None

        if role == "device":
            devices = layout.get("devices") or {}
            if device_class and isinstance(devices, dict):
                per_device = _as_int_list(devices.get(device_class))
                if per_device is not None:
                    return per_device
            return _as_int_list(layout.get("device_default"))

        key_map = {
            "device_starter_main": "device_starter_main",
            "device_manager": "device_manager",
            "policy_runner": "policy_runner",
            "visualizer": "visualizer",
        }
        key = key_map.get(role, role)
        return _as_int_list(layout.get(key))

    def apply(
        self,
        role: RoleName,
        *,
        device_class: Optional[str] = None,
        device_class_id: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return
        n = psutil.cpu_count() or 1
        if n < self.min_cpus:
            suffix = f" {device_class} id={device_class_id}" if device_class is not None else ""
            logger.info(
                f"CPU affinity skipped for {role}{suffix}: "
                f"need >={self.min_cpus} logical CPUs (found {n})"
            )
            return

        cores = self.cores_for_role(role, device_class=device_class)
        if cores is None:
            label = role
            if device_class is not None:
                label = f"{role} {device_class}"
                if device_class_id is not None:
                    label += f" id={device_class_id}"
            logger.info(f"CPU affinity skipped for {label}: no layout entry")
            return

        label = role
        if role == "device" and device_class is not None:
            label = f"device worker {device_class}"
            if device_class_id is not None:
                label += f" id={device_class_id}"
        elif role == "visualizer" and device_class is not None:
            label = f"visualizer {device_class}"
            if device_class_id is not None:
                label += f" id={device_class_id}"

        pin_process_or_warn(cores, role=label)
