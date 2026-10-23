"""Minimal, side-effect-free SuperInference device SHM protocol.

This module intentionally does not import ``utils.shm_utils`` or the repository
logger.  The web process is a read-only consumer; producer creation remains in
the spawned hardware worker.
"""

from __future__ import annotations

import struct
import mmap
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


DTYPE_MAP = {
    'uint8': np.uint8,
    'int8': np.int8,
    'uint16': np.uint16,
    'int16': np.int16,
    'uint32': np.uint32,
    'int32': np.int32,
    'float32': np.float32,
    'float64': np.float64,
}

MAX_SHAPE_DIMENSIONS = 4
DEVICE_HEADER_FORMAT = '<20sIf20sI' + 'I' * MAX_SHAPE_DIMENSIONS + 'If'
BUFFER_HEADER_FORMAT = '<IIII'
FRAME_HEADER_FORMAT = '<Q'
DEVICE_HEADER_SIZE = struct.calcsize(DEVICE_HEADER_FORMAT)
BUFFER_HEADER_SIZE = struct.calcsize(BUFFER_HEADER_FORMAT)
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER_FORMAT)


def _padded_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(shape) > MAX_SHAPE_DIMENSIONS:
        raise ValueError('shape has too many dimensions')
    return shape + (0,) * (MAX_SHAPE_DIMENSIONS - len(shape))


def pack_device_header(
    device_type: str,
    device_class_id: int,
    fps: float,
    data_dtype: str,
    shape: Tuple[int, ...],
    frame_size: int,
    hardware_latency_ms: float = 0.0,
) -> bytes:
    shape = tuple(shape)
    return struct.pack(
        DEVICE_HEADER_FORMAT,
        device_type.encode('utf-8')[:20].ljust(20, b'\x00'),
        int(device_class_id),
        float(fps),
        data_dtype.encode('utf-8')[:20].ljust(20, b'\x00'),
        len(shape),
        *_padded_shape(shape),
        int(frame_size),
        float(hardware_latency_ms),
    )


def unpack_device_header(data: bytes) -> Dict[str, Any]:
    if len(data) < DEVICE_HEADER_SIZE:
        raise ValueError(
            f'Data too short for device header: {len(data)} < {DEVICE_HEADER_SIZE}'
        )
    values = struct.unpack(DEVICE_HEADER_FORMAT, data[:DEVICE_HEADER_SIZE])
    num_dims = int(values[4])
    if num_dims < 0 or num_dims > MAX_SHAPE_DIMENSIONS:
        raise ValueError(f'Invalid device-header dimension count: {num_dims}')
    shape = tuple(values[5 : 5 + MAX_SHAPE_DIMENSIONS])
    return {
        'device_type': values[0].decode('utf-8').rstrip('\x00'),
        'device_class_id': values[1],
        'fps': values[2],
        'data_dtype': values[3].decode('utf-8').rstrip('\x00'),
        'num_dims': num_dims,
        'shape': shape[:num_dims] if num_dims > 0 else (1,),
        'frame_size': values[5 + MAX_SHAPE_DIMENSIONS],
        'hardware_latency_ms': values[6 + MAX_SHAPE_DIMENSIONS],
    }


def pack_buffer_header(
    buffer_size: int, current_frames_count: int, write_index: int
) -> bytes:
    return struct.pack(
        BUFFER_HEADER_FORMAT,
        int(buffer_size),
        int(current_frames_count),
        int(write_index),
        0,
    )


def unpack_buffer_header(data: bytes) -> Dict[str, int]:
    if len(data) < BUFFER_HEADER_SIZE:
        raise ValueError(
            f'Data too short for buffer header: {len(data)} < {BUFFER_HEADER_SIZE}'
        )
    values = struct.unpack(BUFFER_HEADER_FORMAT, data[:BUFFER_HEADER_SIZE])
    return {
        'buffer_size': values[0],
        'current_frames_count': values[1],
        'write_index': values[2],
        'padding': values[3],
    }


def pack_frame_header(timestamp_ns: int) -> bytes:
    return struct.pack(FRAME_HEADER_FORMAT, int(timestamp_ns))


def unpack_frame_header(data: bytes) -> Dict[str, int]:
    if len(data) < FRAME_HEADER_SIZE:
        raise ValueError(
            f'Data too short for frame header: {len(data)} < {FRAME_HEADER_SIZE}'
        )
    return {'timestamp_ns': struct.unpack(FRAME_HEADER_FORMAT, data[:8])[0]}


def calculate_device_shm_size(buffer_size: int, frame_data_size: int) -> int:
    return int(
        DEVICE_HEADER_SIZE
        + BUFFER_HEADER_SIZE
        + int(buffer_size) * (FRAME_HEADER_SIZE + int(frame_data_size))
    )


class ReadOnlyMapping:
    """Read-only `/dev/shm` mapping with no resource-tracker registration."""

    def __init__(self, name: str, fd: int, size: int, mapping: mmap.mmap) -> None:
        self._name = name
        self._fd = fd
        self._size = size
        self._mapping = mapping
        self._closed = False

    @property
    def buf(self) -> memoryview:
        if self._closed:
            raise ValueError('shared-memory mapping is closed')
        return memoryview(self._mapping).toreadonly()

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_read_only(self) -> bool:
        return True

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._mapping.close()
        finally:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
        self._closed = True


def _validated_shm_path(name: str) -> Tuple[str, Path]:
    normalized = str(name).lstrip('/')
    if (
        not normalized
        or normalized in {'.', '..'}
        or '/' in normalized
        or '\\' in normalized
        or '\x00' in normalized
    ):
        raise ValueError('invalid shared-memory name')
    return normalized, Path('/dev/shm') / normalized


def connect_to_shared_memory(name: str) -> ReadOnlyMapping:
    """Open and mmap a POSIX SHM file without `SharedMemory` side effects."""
    normalized, path = _validated_shm_path(name)
    flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0)
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        size = int(os.fstat(fd).st_size)
        if size <= 0:
            raise ValueError(f'shared-memory object is empty: {normalized}')
        mapping = mmap.mmap(fd, length=0, access=mmap.ACCESS_READ)
    except Exception:
        os.close(fd)
        raise
    return ReadOnlyMapping(normalized, fd, size, mapping)


def connect_to_device_shm(
    device_name: str, device_class_id: int, read_only: bool = True
) -> ReadOnlyMapping:
    if read_only is not True:
        raise ValueError('control-panel SHM connections are always read-only')
    return connect_to_shared_memory(f'{device_name}_{int(device_class_id)}_data')


__all__ = [
    'BUFFER_HEADER_SIZE',
    'DEVICE_HEADER_SIZE',
    'DTYPE_MAP',
    'FRAME_HEADER_SIZE',
    'ReadOnlyMapping',
    'calculate_device_shm_size',
    'connect_to_device_shm',
    'pack_buffer_header',
    'pack_device_header',
    'pack_frame_header',
    'unpack_buffer_header',
    'unpack_device_header',
    'unpack_frame_header',
]
