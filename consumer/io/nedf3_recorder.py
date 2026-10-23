"""Small core orchestration layer for NEDF3 episode recording."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .nedf3_contract import NEDF3EpisodeMetadata, NEDF3Writer, create_vendor_writer, utc_timestamp_ns


class NEDF3Recorder:
    """Write typed episode messages without coupling core to the NEDF SDK."""

    def __init__(self, output_path: str, *, topics: Optional[Sequence[str]] = None) -> None:
        self.output_path = output_path
        self.topics = tuple(topics) if topics is not None else None
        self._writer: Optional[NEDF3Writer] = None
        self.metadata: Optional[NEDF3EpisodeMetadata] = None

    def start(self, metadata: NEDF3EpisodeMetadata) -> None:
        if self._writer is not None:
            raise RuntimeError("NEDF3Recorder episode is already active")
        self.metadata = metadata
        kwargs = {"output_path": self.output_path}
        if self.topics is not None:
            kwargs["topics"] = self.topics
        self._writer = create_vendor_writer(**kwargs)

    def write(self, topic: str, message: Any, timestamp_ns: Optional[int] = None) -> None:
        if self._writer is None:
            raise RuntimeError("NEDF3Recorder.start() must be called before write()")
        self._writer.write_message(topic, message, timestamp_ns or utc_timestamp_ns())

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        self._writer = None

    def __enter__(self) -> "NEDF3Recorder":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()


__all__ = ["NEDF3Recorder"]
