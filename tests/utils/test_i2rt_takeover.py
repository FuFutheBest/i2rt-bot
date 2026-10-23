"""Hardware-free contract tests for I2RT takeover hold acknowledgements."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from utils.i2rt_takeover import (
    TAKEOVER_HOLD_ACK_SCHEMA,
    TAKEOVER_HOLD_REQUEST_SCHEMA,
    TakeoverHoldSyncError,
    publish_takeover_hold_ack,
    publish_takeover_hold_request,
    read_takeover_hold_ack,
    read_takeover_hold_request,
)


REQUEST = "a" * 32


def test_hold_request_and_ack_bind_both_devices_and_the_hold_pose(tmp_path: Path) -> None:
    request_path = tmp_path / "takeover-request.json"
    acknowledgement_path = tmp_path / "takeover-ack.json"
    hold_pose = np.arange(14, dtype=np.float64) / 10.0

    request_id = publish_takeover_hold_request(
        request_path,
        source_device_id=0,
        target_device_id=1,
        request_id=REQUEST.upper(),
    )
    publish_takeover_hold_ack(
        acknowledgement_path,
        request_id=request_id,
        source_device_id=1,
        target_device_id=0,
        hold_joint_pos=hold_pose,
        measured_at_ns=123,
    )

    request = json.loads(request_path.read_text(encoding="utf-8"))
    acknowledgement = json.loads(acknowledgement_path.read_text(encoding="utf-8"))
    assert request["schema_version"] == TAKEOVER_HOLD_REQUEST_SCHEMA
    assert acknowledgement["schema_version"] == TAKEOVER_HOLD_ACK_SCHEMA
    assert acknowledgement["measured_at_ns"] == 123
    assert (
        read_takeover_hold_request(
            request_path,
            expected_source_device_id=0,
            expected_target_device_id=1,
        )
        == REQUEST
    )
    np.testing.assert_allclose(
        read_takeover_hold_ack(
            acknowledgement_path,
            request_id=REQUEST,
            expected_source_device_id=1,
            expected_target_device_id=0,
        ),
        hold_pose,
    )


def test_hold_ack_ignores_stale_request_but_rejects_wrong_source(tmp_path: Path) -> None:
    acknowledgement_path = tmp_path / "takeover-ack.json"
    publish_takeover_hold_ack(
        acknowledgement_path,
        request_id="b" * 32,
        source_device_id=1,
        target_device_id=0,
        hold_joint_pos=np.zeros(14),
        measured_at_ns=123,
    )

    assert (
        read_takeover_hold_ack(
            acknowledgement_path,
            request_id=REQUEST,
            expected_source_device_id=1,
            expected_target_device_id=0,
        )
        is None
    )

    publish_takeover_hold_ack(
        acknowledgement_path,
        request_id=REQUEST,
        source_device_id=2,
        target_device_id=0,
        hold_joint_pos=np.zeros(14),
        measured_at_ns=124,
    )
    with pytest.raises(TakeoverHoldSyncError, match="source_device_id"):
        read_takeover_hold_ack(
            acknowledgement_path,
            request_id=REQUEST,
            expected_source_device_id=1,
            expected_target_device_id=0,
        )


def test_hold_request_rejects_an_unexpected_target_device(tmp_path: Path) -> None:
    request_path = tmp_path / "takeover-request.json"
    publish_takeover_hold_request(
        request_path,
        source_device_id=0,
        target_device_id=1,
        request_id=REQUEST,
    )

    with pytest.raises(TakeoverHoldSyncError, match="target_device_id"):
        read_takeover_hold_request(
            request_path,
            expected_source_device_id=0,
            expected_target_device_id=0,
        )
