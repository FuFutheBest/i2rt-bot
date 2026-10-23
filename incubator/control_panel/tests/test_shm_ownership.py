"""POSIX shared-memory ownership regression tests (no hardware required)."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
import uuid

import pytest

from incubator.control_panel import lifecycle
from utils import shm_utils


def _shm_path(name: str) -> Path:
    return Path('/dev/shm') / name.lstrip('/')


def _replace_in_other_process(
    name: str, marker: bytes, connection: object
) -> None:
    """Replace a claimed name and keep the replacement alive for inspection."""
    connection.recv()
    original = shared_memory.SharedMemory(name=name)
    original.close()
    original.unlink()

    replacement = shared_memory.SharedMemory(create=True, size=128, name=name)
    replacement.buf[: len(marker)] = marker
    stat_result = _shm_path(name).stat()
    connection.send((int(stat_result.st_dev), int(stat_result.st_ino)))
    connection.recv()
    connection.send(bytes(replacement.buf[: len(marker)]))
    replacement.close()
    replacement.unlink()
    connection.close()


def _hold_external_flag(name: str, marker: bytes, connection: object) -> None:
    external = shared_memory.SharedMemory(create=True, size=128, name=name)
    external.buf[: len(marker)] = marker
    stat_result = _shm_path(name).stat()
    connection.send((int(stat_result.st_dev), int(stat_result.st_ino)))
    connection.recv()
    current = _shm_path(name).stat()
    connection.send(
        (
            int(current.st_dev),
            int(current.st_ino),
            bytes(external.buf[: len(marker)]),
        )
    )
    connection.recv()
    external.close()
    external.unlink()
    connection.close()


@pytest.fixture(autouse=True)
def exclusive_creation_hook() -> None:
    original = shm_utils.create_shared_memory
    lifecycle._WORKER_OWNED_SHARED_MEMORY.clear()
    lifecycle._WORKER_ORIGINAL_SHARED_MEMORY = shared_memory.SharedMemory
    lifecycle._WORKER_CLAIM_CONFLICT = None
    shm_utils.create_shared_memory = lifecycle._exclusive_worker_create_shared_memory
    try:
        yield
    finally:
        shm_utils.create_shared_memory = original
        lifecycle._cleanup_worker_owned_shared_memory()
        lifecycle._WORKER_ORIGINAL_SHARED_MEMORY = None
        lifecycle._WORKER_CLAIM_CONFLICT = None


@pytest.mark.skipif(not Path('/dev/shm').is_dir(), reason='POSIX SHM required')
def test_device_and_control_helpers_use_exclusive_hook_and_restart_cleanly() -> None:
    """Both imported helper aliases resolve the patched shm_utils global."""
    device_name = f'ControlPanelTest_{uuid.uuid4().hex}'
    data_name = f'{device_name}_17_data'
    control_name = f'{device_name}_17_control'

    for _ in range(2):
        data = shm_utils.create_device_shm(device_name, 17, 2, 16)
        control = shm_utils.create_control_shm(device_name, 17, 2, 16)
        assert lifecycle._worker_make_readonly(control) is control
        assert control.is_read_only is True
        assert _shm_path(data_name).is_file()
        assert _shm_path(control_name).is_file()
        assert {claim['name'] for claim in lifecycle._WORKER_OWNED_SHARED_MEMORY} == {
            data_name,
            control_name,
        }
        data.close()
        data.unlink()
        control.close()
        control.unlink()
        lifecycle._cleanup_worker_owned_shared_memory()
        assert not _shm_path(data_name).exists()
        assert not _shm_path(control_name).exists()


@pytest.mark.skipif(not Path('/dev/shm').is_dir(), reason='POSIX SHM required')
def test_secondary_conflict_cleans_primary_and_preserves_external_secondary() -> None:
    device_name = f'ControlPanelTest_{uuid.uuid4().hex}'
    data_name = f'{device_name}_23_data'
    control_name = f'{device_name}_23_control'
    external = shared_memory.SharedMemory(create=True, size=128, name=control_name)
    marker = b'external-secondary-owner'
    external.buf[: len(marker)] = marker
    external_identity = _shm_path(control_name).stat()

    try:
        shm_utils.create_device_shm(device_name, 23, 2, 16)
        with pytest.raises(FileExistsError):
            shm_utils.create_control_shm(device_name, 23, 2, 16)
        lifecycle._cleanup_worker_owned_shared_memory()

        assert not _shm_path(data_name).exists()
        current = _shm_path(control_name).stat()
        assert (current.st_dev, current.st_ino) == (
            external_identity.st_dev,
            external_identity.st_ino,
        )
        assert bytes(external.buf[: len(marker)]) == marker
    finally:
        external.close()
        external.unlink()


@pytest.mark.skipif(not Path('/dev/shm').is_dir(), reason='POSIX SHM required')
def test_cleanup_does_not_unlink_recreated_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = f'ControlPanelTest_{uuid.uuid4().hex}'
    marker = b'replacement-owner'
    channel = lifecycle._exclusive_worker_create_shared_memory(name, 128)
    claim = lifecycle._WORKER_OWNED_SHARED_MEMORY[-1]
    claimed_identity = (claim['dev'], claim['ino'])

    context = mp.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(
        target=_replace_in_other_process,
        args=(name, marker, child),
    )
    process.start()
    child.close()

    real_identity = lifecycle._shared_memory_identity

    def stale_identity(_: str):
        # Simulate the strongest TOCTOU case: cleanup observes the claimed
        # inode, another process replaces the name, then cleanup resumes with
        # that stale answer. Registry finalization must still never unlink.
        parent.send('replace')
        replacement_identity = parent.recv()
        assert replacement_identity != claimed_identity
        return claimed_identity

    monkeypatch.setattr(lifecycle, '_shared_memory_identity', stale_identity)

    lifecycle._cleanup_worker_owned_shared_memory()
    monkeypatch.setattr(lifecycle, '_shared_memory_identity', real_identity)
    parent.send('inspect')
    assert parent.recv() == marker
    process.join(timeout=10.0)
    parent.close()

    assert process.exitcode == 0
    assert not _shm_path(name).exists()
    # The registry closes the original raw handle; the wrapper must not be used
    # again after ownership cleanup.
    del channel


@pytest.mark.parametrize('direct_path', ['teleoperator', 'robot'])
@pytest.mark.skipif(not Path('/dev/shm').is_dir(), reason='POSIX SHM required')
def test_direct_shared_control_preflight_preserves_external_flag(
    direct_path: str,
) -> None:
    """Model both direct SharedMemory cleanup-existing creator paths."""
    name = f'ControlPanel_{direct_path}_{uuid.uuid4().hex}_control_flag'
    marker = f'external-{direct_path}'.encode('ascii')
    context = mp.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(target=_hold_external_flag, args=(name, marker, child))
    process.start()
    child.close()
    original_identity = parent.recv()

    attached = lifecycle._guarded_direct_shared_memory(name=name, create=False)
    attached.close()
    # The BaseException-derived conflict intentionally escapes the legacy
    # method's broad `except Exception`, aborting start_server before an
    # external flag can be removed or silently ignored.
    with pytest.raises(lifecycle._DirectSharedMemoryConflict):
        attached.unlink()

    assert lifecycle._WORKER_CLAIM_CONFLICT == name
    parent.send('inspect')
    current_dev, current_ino, current_marker = parent.recv()
    assert (current_dev, current_ino) == original_identity
    assert current_marker == marker
    parent.send('cleanup')
    process.join(timeout=10.0)
    parent.close()
    assert process.exitcode == 0


@pytest.mark.skipif(not Path('/dev/shm').is_dir(), reason='POSIX SHM required')
def test_direct_owned_control_flag_uses_normal_owner_cleanup() -> None:
    name = f'ControlPanel_owned_{uuid.uuid4().hex}_control_flag'
    channel = lifecycle._guarded_direct_shared_memory(
        name=name,
        create=True,
        size=128,
    )
    channel.buf[:4] = b'flag'
    assert bytes(channel.buf[:4]) == b'flag'
    channel.close()
    channel.unlink()
    lifecycle._cleanup_worker_owned_shared_memory()
    assert not _shm_path(name).exists()
