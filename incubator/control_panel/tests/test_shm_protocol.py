"""Side-effect-free read-only POSIX SHM mapping tests."""

from __future__ import annotations

from multiprocessing import shared_memory
from pathlib import Path
import uuid

import pytest

from incubator.control_panel.shm_protocol import connect_to_shared_memory


@pytest.mark.skipif(not Path('/dev/shm').is_dir(), reason='POSIX SHM required')
def test_read_only_mapping_preserves_producer_segment() -> None:
    name = f'ControlPanelMapping_{uuid.uuid4().hex}'
    producer = shared_memory.SharedMemory(create=True, size=64, name=name)
    marker = b'producer-owned-data'
    producer.buf[: len(marker)] = marker
    try:
        reader = connect_to_shared_memory(name)
        view = reader.buf
        assert view.readonly is True
        assert bytes(view[: len(marker)]) == marker
        with pytest.raises(TypeError):
            view[0] = 0
        view.release()
        reader.close()

        attached = shared_memory.SharedMemory(name=name)
        try:
            assert bytes(attached.buf[: len(marker)]) == marker
        finally:
            attached.close()
    finally:
        producer.close()
        producer.unlink()


@pytest.mark.parametrize('name', ['', '../escape', 'nested/name', '..'])
def test_mapping_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError):
        connect_to_shared_memory(name)
