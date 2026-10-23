"""
Inference profiling helpers for SuperInference server and policy processes.

Author: SuperInference Team
"""

from __future__ import annotations

import copy
import json
import os
import socket
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, ContextManager, Optional

from utils.inference_options import as_bool
from utils.logger_config import logger


class ResourceMonitor:
    """Persist lightweight process and CUDA allocator snapshots as JSONL."""

    def __init__(
        self,
        enabled: bool,
        output_root: str,
        run_id: str,
        component: str,
    ) -> None:
        self.enabled = enabled
        self.component = component
        self._path = (
            Path(output_root)
            / run_id
            / socket.gethostname()
            / "rank_0"
            / "resources"
            / f"{component}.jsonl"
        )

    @staticmethod
    def _snapshot() -> dict[str, Any]:
        page_size = os.sysconf("SC_PAGE_SIZE")
        with Path("/proc/self/statm").open(encoding="utf-8") as file:
            resident_pages = int(file.read().split()[1])
        snapshot: dict[str, Any] = {
            "wall_time_ns": time.time_ns(),
            "process_cpu_seconds": time.process_time(),
            "rss_bytes": resident_pages * page_size,
        }
        try:
            import torch

            if torch.cuda.is_available():
                snapshot.update(
                    {
                        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
                        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    }
                )
        except Exception:
            pass
        return snapshot

    def record(self, event: str, step: int, section: str) -> None:
        """Append one resource snapshot without affecting inference on failure."""
        if not self.enabled:
            return
        try:
            record = {
                "component": self.component,
                "event": event,
                "step": step,
                "section": section,
                **self._snapshot(),
            }
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as exc:
            logger.debug(f"Failed to write inference resource snapshot: {exc}")


class InferenceProfiler:
    """No-op-safe nmx-profiler and resource-monitor integration."""

    def __init__(
        self,
        raw_config: Optional[Mapping[str, Any]],
        component: str,
    ) -> None:
        self.component = component
        self._step = -1
        config = self._resolve_component_config(raw_config, component)
        self._profiler = self._create_profiler(config)
        output = dict(config.get("output", {}))
        if self._profiler is not None and os.environ.get("NMX_PROFILE_CONFIG"):
            output = {
                "root": self._profiler.config.output.root,
                "run_id": self._profiler.config.output.run_id,
            }
        self._resources = ResourceMonitor(
            enabled=as_bool(config.get("resources", {}).get("enabled", False))
            or as_bool(os.environ.get("NMX_INFERENCE_RESOURCES", False)),
            output_root=str(output.get("root", "results/profiles")),
            run_id=str(output.get("run_id", f"fastwam-{component}")),
            component=component,
        )

    @staticmethod
    def _resolve_component_config(
        raw_config: Optional[Mapping[str, Any]],
        component: str,
    ) -> dict[str, Any]:
        root = copy.deepcopy(dict(raw_config or {}))
        component_config = root.pop(component, {})
        if not isinstance(component_config, Mapping):
            raise TypeError(f"profiling.{component} must be a mapping")
        root.pop("server", None)
        root.pop("client", None)
        root.update(component_config)
        return root

    @staticmethod
    def _create_profiler(config: Mapping[str, Any]) -> Any:
        try:
            from nmx_profiler import TrainingProfiler
        except Exception as exc:
            if str(config.get("mode", "off")) != "off":
                logger.warning(f"nmx-profiler is unavailable; profiling disabled: {exc}")
            return None

        try:
            if os.environ.get("NMX_PROFILE_CONFIG"):
                return TrainingProfiler.from_env()
            profiler_config = dict(config)
            profiler_config.pop("resources", None)
            return TrainingProfiler.from_config(profiler_config)
        except Exception as exc:
            logger.warning(f"Failed to initialize nmx-profiler: {exc}")
            return None

    @property
    def enabled(self) -> bool:
        """Return whether nmx-profiler has an active profiling mode."""
        return bool(self._profiler is not None and getattr(self._profiler, "enabled", False))

    def step(self, step: int) -> None:
        """Advance the profiler once for each completed inference request."""
        self._step = int(step)
        if self._profiler is not None:
            self._profiler.step(self._step)

    @contextmanager
    def section(
        self,
        name: str,
        category: str = "coarse",
    ) -> Iterator[None]:
        """Measure a named section and write before/after resource snapshots."""
        self._resources.record("start", self._step, name)
        context: ContextManager[None] = (
            self._profiler.section(name, {"category": category})
            if self._profiler is not None
            else nullcontext()
        )
        try:
            with context:
                yield
        finally:
            self._resources.record("end", self._step, name)

    def finalize(self) -> None:
        """Flush nmx-profiler artifacts once during normal process shutdown."""
        if self._profiler is None:
            return
        try:
            self._profiler.finalize()
        except Exception as exc:
            logger.warning(f"Failed to finalize {self.component} profiler: {exc}")
