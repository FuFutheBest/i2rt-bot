"""Shared-memory ownership helpers specific to the RoboTwin bridge."""

from __future__ import annotations

from multiprocessing import shared_memory
from typing import Any, MutableMapping, Sequence


def close_and_unlink_owned_channels(
    channels: MutableMapping[str, Any],
    names: Sequence[str] | None,
) -> None:
    """Close read-only views, then unlink SHMs that this bridge created.

    ``BaseRobot`` wraps its newly-created control SHM in a read-only
    ``DataChannel``. That view intentionally rejects ``unlink()``, so a bridge
    that owns the name must reopen it as a normal ``SharedMemory`` owner during
    shutdown.
    """
    for name in names or ():
        channel = channels.get(name)
        if channel is not None:
            try:
                channel.close()
            except (BufferError, FileNotFoundError, OSError):
                pass
        try:
            owner = shared_memory.SharedMemory(name=name, create=False)
        except FileNotFoundError:
            owner = None
        if owner is not None:
            try:
                owner.unlink()
            except FileNotFoundError:
                pass
            finally:
                owner.close()
        channels[name] = None
