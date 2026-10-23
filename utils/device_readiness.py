"""Shared-memory based device startup readiness checks."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from utils.logger_config import logger
from utils.shm_utils import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    connect_to_shared_memory,
    unpack_buffer_header,
    unpack_device_header,
    unpack_frame_header,
)

DEFAULT_DEVICE_READY_TIMEOUT_S = 40.0
DEFAULT_DEVICE_READY_POLL_INTERVAL_S = 0.1


class DeviceReadinessError(RuntimeError):
    """Raised when a configured device cannot prove live stream readiness."""


def _device_key(device_info: Mapping[str, Any]) -> str:
    return f"{device_info['device_class']}_{device_info['device_class_id']}"


def _stream_name(device_info: Mapping[str, Any]) -> str:
    device_name = (
        device_info.get("config", {}).get("device_name")
        or device_info["device_class"]
    )
    return f"{device_name}_{device_info['device_class_id']}_data"


def _latest_frame_signature(
    channel: Any,
    device_info: Mapping[str, Any],
) -> tuple[int, int, int] | None:
    buffer = channel.buf
    device_header = unpack_device_header(bytes(buffer[:DEVICE_HEADER_SIZE]))
    expected_class = str(
        device_info.get("config", {}).get("device_name")
        or device_info["device_class"]
    )
    expected_id = int(device_info["device_class_id"])
    if (
        device_header["device_type"] != expected_class
        or device_header["device_class_id"] != expected_id
    ):
        raise ValueError(
            "device stream header does not match its configured device: "
            f"expected {expected_class}_{expected_id}, got "
            f"{device_header['device_type']}_{device_header['device_class_id']}"
        )

    buffer_header = unpack_buffer_header(
        bytes(buffer[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE])
    )
    buffer_size = int(buffer_header["buffer_size"])
    frame_count = int(buffer_header["current_frames_count"])
    write_index = int(buffer_header["write_index"])
    if buffer_size <= 0:
        raise ValueError("device stream has an invalid buffer size")
    if frame_count <= 0:
        return None

    frame_size = int(device_header["frame_size"])
    if frame_size < FRAME_HEADER_SIZE:
        raise ValueError("device stream has an invalid frame size")
    latest_index = (write_index - 1) % buffer_size
    frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + latest_index * frame_size
    timestamp_ns = int(
        unpack_frame_header(
            bytes(buffer[frame_offset : frame_offset + FRAME_HEADER_SIZE])
        )["timestamp_ns"]
    )
    if timestamp_ns <= 0:
        return None
    return frame_count, write_index, timestamp_ns


def _process_is_alive(process: Any) -> bool:
    if process is None:
        return False
    try:
        return bool(process.is_alive())
    except (AssertionError, ValueError):
        return False


def wait_for_device_streams(
    devices: Sequence[Mapping[str, Any]],
    *,
    timeout_s: float = DEFAULT_DEVICE_READY_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_DEVICE_READY_POLL_INTERVAL_S,
    connect_fn: Callable[..., Any] | None = None,
) -> None:
    """Wait until every configured device publishes two distinct valid frames.

    Requiring a changing frame signature proves that the producer is live and
    avoids accepting a stale shared-memory segment left by an earlier process.
    """
    if not devices:
        return

    connect = connect_fn or connect_to_shared_memory
    owner_by_name = {_device_key(device): device for device in devices}
    pending = list(devices)
    first_signatures: dict[str, tuple[int, int, int]] = {}
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    poll_interval_s = max(0.01, float(poll_interval_s))
    logger.info(
        f"Waiting up to {max(0.0, float(timeout_s)):.1f}s for "
        f"{len(pending)} device stream(s) to publish live frames"
    )

    while pending:
        still_pending: list[Mapping[str, Any]] = []
        for device_info in pending:
            managed_by = device_info.get("config", {}).get("managed_by")
            owner = owner_by_name.get(str(managed_by)) if managed_by else device_info
            stream_name = _stream_name(device_info)
            if owner is None:
                raise DeviceReadinessError(
                    f"Device stream {stream_name} names unknown owner {managed_by!r}"
                )
            if not _process_is_alive(owner.get("process")):
                raise DeviceReadinessError(
                    f"Device stream owner {_device_key(owner)} exited before "
                    f"{stream_name} became ready"
                )

            try:
                channel = connect(
                    stream_name,
                    read_only=True,
                    log_missing=False,
                )
            except FileNotFoundError:
                still_pending.append(device_info)
                continue
            except Exception as exc:
                logger.debug(f"Cannot open device stream {stream_name} yet: {exc}")
                still_pending.append(device_info)
                continue

            try:
                signature = _latest_frame_signature(channel, device_info)
            except Exception as exc:
                logger.debug(f"Cannot inspect device stream {stream_name} yet: {exc}")
                still_pending.append(device_info)
            else:
                first_signature = first_signatures.get(stream_name)
                if signature is None:
                    still_pending.append(device_info)
                elif first_signature is None:
                    first_signatures[stream_name] = signature
                    still_pending.append(device_info)
                elif signature == first_signature:
                    still_pending.append(device_info)
                else:
                    logger.info(f"Device stream ready: {stream_name}")
            finally:
                channel.close()

        if not still_pending:
            return
        if time.monotonic() >= deadline:
            names = ", ".join(_stream_name(device) for device in still_pending)
            raise DeviceReadinessError(
                f"Timed out waiting for live device stream(s): {names}"
            )

        pending = still_pending
        time.sleep(poll_interval_s)
