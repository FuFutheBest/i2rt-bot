#!/usr/bin/env python3
"""
Backend controller for the Vive_DM3 combined GUI.

Author: Yichen Xie
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from incubator.nedf3.Vive_DM3.calibration_manager import CalibrationManager
from incubator.nedf3.Vive_DM3.metadata_helpers import expand_triple_csv
from incubator.nedf3.Vive_DM3.process_manager import (
    DEFAULT_CONFIG_NAME,
    ProcessManager,
    load_config_defaults,
    save_camera_capture_indices,
    save_encoder_ports,
    scan_config_files,
)
from incubator.nedf3.Vive_DM3.session import SessionStore
from incubator.nedf3.Vive_DM3.upload_manager import AutoUploadManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "my_task"
DEFAULT_TASK_NAME = "my_task"
DEFAULT_EXTRA_DEVICE_TYPE = "robopocket_2.0_pro"
DEFAULT_EXTRA_HARDWARE_STRING = "310"

class ViveDM3Controller:
    """Shared backend for DM3 login, task selection, collection, and upload manager creation."""

    def __init__(self) -> None:
        self.session = SessionStore()
        self.processes = ProcessManager()
        self.calibration = CalibrationManager(lambda: self.processes.device.running)

        self.config_name = DEFAULT_CONFIG_NAME
        self.camera_indices: Dict[str, str] = {}
        self.encoder_ports: Dict[str, str] = {"left_port": "", "right_port": ""}
        self.output_dir = str(DEFAULT_OUTPUT_DIR)
        self.task_name = DEFAULT_TASK_NAME
        self.resource_type = "robopocket20_pro"
        self.extra_device_type = DEFAULT_EXTRA_DEVICE_TYPE
        self.extra_hardware_string = DEFAULT_EXTRA_HARDWARE_STRING
        self.extra_device_zero = ""
        self._upload_manager: Optional[AutoUploadManager] = None
        self.reload_config_defaults()

    def reload_config_defaults(self) -> None:
        defaults = load_config_defaults(self.config_name)
        self.camera_indices = dict(defaults.get("camera_indices") or {})
        self.encoder_ports = dict(defaults.get("encoder_ports") or {})

    def apply_selected_task_defaults(self) -> None:
        state = self.session.public_state()
        task_name = str(state.get("task_name") or DEFAULT_TASK_NAME).strip() or DEFAULT_TASK_NAME
        self.task_name = task_name
        self.output_dir = str(DEFAULT_DATA_ROOT / task_name)

    def save_encoder_ports_to_config(self) -> None:
        save_encoder_ports(self.config_name, self.encoder_ports)

    def save_camera_indices_to_config(self) -> None:
        """Persist USB capture_id values into the active Hydra config YAML."""
        save_camera_capture_indices(self.config_name, self.camera_indices)

    def start_device(self) -> None:
        self.processes.start_device(
            config_name=self.config_name,
            camera_indices=self.camera_indices,
            encoder_ports=self.encoder_ports,
        )

    def start_saver(self) -> None:
        session = self.session.require_ready_for_collection()
        metadata = session.metadata_for_saver()
        if not self.extra_device_zero.strip():
            raise ValueError("请填写 device identifier 设备编号。")
        self.processes.start_saver(
            config_name=self.config_name,
            output_dir=self.output_dir,
            task_name=self.task_name,
            extra_device_type=expand_triple_csv(
                self.extra_device_type,
                default=DEFAULT_EXTRA_DEVICE_TYPE,
            ),
            extra_hardware_string=expand_triple_csv(
                self.extra_hardware_string,
                default=DEFAULT_EXTRA_HARDWARE_STRING,
            ),
            extra_device_zero=expand_triple_csv(self.extra_device_zero),
            dm3_user_id=metadata["user_id"],
            dm3_user_name=metadata["user_name"],
            task_description=metadata["task_description"],
        )

    def create_upload_manager(
        self,
        data_dir: Path,
        log_callback=None,
    ) -> AutoUploadManager:
        """Create or reuse the unified episode upload manager."""

        def fetch_sts_token(resource_type: str) -> Dict[str, Any]:
            client = self.session.get_client()
            return client.fetch_oss_sts_token(resource_type=resource_type)

        resolved_dir = data_dir.expanduser().resolve()
        if self._upload_manager is None:
            self._upload_manager = AutoUploadManager(
                data_dir=resolved_dir,
                token_fetcher=fetch_sts_token,
                log_callback=log_callback,
            )
        else:
            self._upload_manager.configure(data_dir=resolved_dir)
            self._upload_manager.log_callback = log_callback
        self._upload_manager.configure(resource_type=self.resource_type)
        return self._upload_manager

    def shutdown_upload_manager(self) -> None:
        if self._upload_manager is None:
            return
        self._upload_manager.shutdown()
        self._upload_manager = None

    def shutdown(self) -> None:
        if self.processes.saver.running:
            self.processes.stop_saver()
        if self.processes.device.running:
            self.processes.stop_device()
        if self.calibration.status_dict().get("active"):
            self.calibration.stop()
        self.shutdown_upload_manager()
        self.session.logout()
