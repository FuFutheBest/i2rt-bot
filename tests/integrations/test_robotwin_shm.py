"""Shared-memory cleanup contract for the RoboTwin bridge control channel."""

from __future__ import annotations

from multiprocessing import shared_memory

import pytest

from integrations.robotwin.shm import close_and_unlink_owned_channels
from utils.shm_utils import make_readonly


def test_cleanup_unlinks_a_readonly_control_channel_owned_by_bridge() -> None:
    owner = shared_memory.SharedMemory(create=True, size=64)
    name = owner.name
    channels = {name: make_readonly(owner)}

    close_and_unlink_owned_channels(channels, [name])

    assert channels[name] is None
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name, create=False)
