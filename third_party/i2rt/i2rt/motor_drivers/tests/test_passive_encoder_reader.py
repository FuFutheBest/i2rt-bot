"""Tests for passive teaching-handle encoder normalization."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from i2rt.motor_drivers.dm_driver import PassiveEncoderReader, ReceiveMode


class _FakeCanInterface:
    def _send_message_get_response(
        self, *args, **kwargs  # noqa: ANN002, ANN003
    ) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace()


def _reader(raw_position: float) -> PassiveEncoderReader:
    reader = PassiveEncoderReader.__new__(PassiveEncoderReader)
    reader.can_interface = _FakeCanInterface()
    reader.range_rad = 0.7
    reader.receive_mode = ReceiveMode.plus_one
    reader._parse_encoder_message = lambda _message: (raw_position, 0.0, [False, False])
    return reader


@pytest.mark.parametrize(
    ("raw_position", "expected_position"),
    [
        (0.0, 0.0),
        (0.35, 0.5),
        (-0.35, 0.5),
        (2 * math.pi - 0.01, 0.01 / 0.7),
        (2 * math.pi + 0.01, 0.01 / 0.7),
        (1.0, 1.0),
    ],
)
def test_read_encoder_wraps_angle_before_normalizing(
    raw_position: float, expected_position: float
) -> None:
    state = _reader(raw_position).read_encoder(0x50E)

    assert state.position == pytest.approx(expected_position)
