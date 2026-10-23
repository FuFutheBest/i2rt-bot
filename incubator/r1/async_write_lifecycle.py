"""Reusable queue and thread lifecycle for asynchronous data writers."""

import queue
import threading
import time
from typing import Any, Dict, Optional

from utils.logger_config import logger


_WRITE_QUEUE_POLL_INTERVAL_S = 0.1


class AsyncWriteLifecycleMixin:
    """Provide sentinel-based startup, drain, and shutdown for saver workers."""

    def _initialize_async_write_lifecycle(
        self,
        queue_size: int,
        writer_shutdown_timeout_s: float,
    ) -> None:
        queue_capacity = int(queue_size)
        if queue_capacity < 1:
            raise ValueError("queue_size must be >= 1")
        timeout_s = float(writer_shutdown_timeout_s)
        if timeout_s <= 0.0:
            raise ValueError("writer_shutdown_timeout_s must be greater than 0")
        self.write_queue = queue.Queue(maxsize=queue_capacity)
        self.write_thread: Optional[threading.Thread] = None
        self.write_thread_running = False
        self.write_accepting = False
        self.writer_shutdown_timeout_s = timeout_s
        self._write_stop_sentinel = object()

    def _get_next_write_task(self) -> Optional[Dict[str, Any]]:
        """Return the next task, or ``None`` after consuming the stop sentinel."""
        write_task = self.write_queue.get(timeout=_WRITE_QUEUE_POLL_INTERVAL_S)
        if write_task is self._write_stop_sentinel:
            self.write_queue.task_done()
            return None
        return write_task

    def _wait_for_write_queue(self, timeout_s: Optional[float] = None) -> bool:
        """Wait for queued tasks without hanging forever after worker failure."""
        timeout_s = self.writer_shutdown_timeout_s if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + timeout_s
        while self.write_queue.unfinished_tasks:
            write_thread = self.write_thread
            if write_thread is None or not write_thread.is_alive():
                logger.error(
                    f"Async writer stopped with {self.write_queue.unfinished_tasks} "
                    "unfinished queue items"
                )
                return False
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                logger.error(
                    f"Timed out after {timeout_s:.1f}s waiting for "
                    f"{self.write_queue.unfinished_tasks} queued writes"
                )
                return False
            time.sleep(min(_WRITE_QUEUE_POLL_INTERVAL_S, remaining_s))
        return True

    def start_async_writer(self) -> None:
        """Start the saver-specific background worker."""
        if self.async_write and not self.write_thread_running:
            self.write_thread_running = True
            self.write_accepting = True
            self.write_thread = threading.Thread(
                target=self._async_write_worker,
                daemon=True,
            )
            self.write_thread.start()
            logger.info("Started async write thread")

    def _on_async_write_shutdown_failure(self, reason: str) -> None:
        """Optional saver hook for marking the current episode incomplete."""

    def stop_async_writer(self) -> None:
        """Drain queued tasks, stop the worker, then close active streams."""
        self.write_accepting = False
        write_thread = self.write_thread
        if write_thread is None:
            if self.episode_active:
                self.close_episode_streams()
            return

        logger.info("Stopping async write thread after draining queued data...")
        deadline = time.monotonic() + self.writer_shutdown_timeout_s
        sentinel_enqueued = False
        while write_thread.is_alive() and time.monotonic() < deadline:
            try:
                remaining_s = max(0.0, deadline - time.monotonic())
                self.write_queue.put(
                    self._write_stop_sentinel,
                    timeout=min(_WRITE_QUEUE_POLL_INTERVAL_S, remaining_s),
                )
                sentinel_enqueued = True
                break
            except queue.Full:
                continue

        if sentinel_enqueued:
            write_thread.join(timeout=max(0.0, deadline - time.monotonic()))

        if write_thread.is_alive():
            reason = (
                f"Async writer shutdown timed out after "
                f"{self.writer_shutdown_timeout_s:.1f}s"
            )
            logger.error(reason)
            self._on_async_write_shutdown_failure(reason)
            return

        self.write_thread_running = False
        self.write_thread = None
        if self.write_queue.unfinished_tasks:
            reason = (
                f"Async writer stopped with {self.write_queue.unfinished_tasks} "
                "unfinished queue items"
            )
            logger.error(reason)
            self._on_async_write_shutdown_failure(reason)
            if self.episode_active:
                self.close_episode_streams()
            return

        logger.info("Async write queue drained and worker stopped")
        if self.episode_active:
            self.close_episode_streams()
