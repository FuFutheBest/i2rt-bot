"""Focused tests for inference-rollout NEDF3 device mappings."""

import json
from typing import Any

import numpy as np
import pytest

_r1_saver = pytest.importorskip(
    "incubator.r1.async_data_saver_NEDF3_r1",
    reason="NEDF3 rollout mapping tests require optional recorder dependencies",
)
R1DataSaverConsumer = _r1_saver.R1DataSaverConsumer


def test_r1_saver_project_root_is_repository_root() -> None:
    saver_path = (
        _r1_saver.PROJECT_ROOT / "incubator" / "r1" / "async_data_saver_NEDF3_r1.py"
    )
    assert saver_path.is_file()
    assert (_r1_saver.PROJECT_ROOT / "configs").is_dir()


def test_openpi_camera_video_ids_map_to_canonical_wrist_roles() -> None:
    saver = R1DataSaverConsumer.__new__(R1DataSaverConsumer)
    saver.devices = {
        "WebCameraOpenPi_1": {"type": "WebCameraOpenPi"},
        "WebCameraOpenPi_3": {"type": "WebCameraOpenPi"},
    }

    entries = saver._camera_entries()

    assert [(index, key) for index, key, _ in entries] == [
        (0, "WebCameraOpenPi_3"),
        (1, "WebCameraOpenPi_1"),
    ]


def test_rollout_control_shms_write_commands_and_predicted_chunks() -> None:
    class _Writer:
        def __init__(self) -> None:
            self.messages: list[tuple[str, Any, int, int]] = []

        def write_message(self, *args: Any) -> None:
            self.messages.append(args)

    saver = R1DataSaverConsumer.__new__(R1DataSaverConsumer)
    saver.nedf_writer = _Writer()

    assert saver._write_control_shm("RizonRobot_2_control", np.arange(16), 123)
    assert [message[0] for message in saver.nedf_writer.messages] == [
        "nmx/hal/arm/left/tcp/cmd",
        "nmx/hal/hand/left/cmd",
        "nmx/hal/arm/right/tcp/cmd",
        "nmx/hal/hand/right/cmd",
    ]

    saver.nedf_writer.messages.clear()
    assert saver._write_control_shm("action_chunk", np.zeros((48, 16)), 456)
    assert saver.nedf_writer.messages[0][0] == "nmx/hal/arm/left/extra"
    payload = json.loads(saver.nedf_writer.messages[0][1].metadata)
    assert payload["policy_action_chunk"]["shape"] == [48, 16]
