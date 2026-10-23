"""Fail-closed GPU checks for inference services.

The helpers only inspect the driver and PyTorch availability; they never
allocate a CUDA tensor.  A process-local fault latch prevents a detected
driver failure from turning into repeated model calls.
"""

from __future__ import annotations

import csv
import os
import subprocess
import threading
import time
from typing import Any, Dict, Optional

DEFAULT_MIN_FREE_MIB = 12_288
_GPU_ERROR_MARKERS = (
    "cuda",
    "cublas",
    "cudnn",
    "nccl",
    "out of memory",
    "out-of-memory",
    "xid",
    "unspecified launch failure",
    "illegal memory access",
    "device-side assert",
    "no kernel image",
    "launch failed",
    "misaligned address",
    "gpu has fallen off the bus",
    "driver",
    "nvml",
    "device is busy",
    "device busy",
    "device unavailable",
    "device is unavailable",
)


def is_cuda_device(device: Any) -> bool:
    return str(device or "").strip().lower().startswith("cuda")


def classify_gpu_error(error: BaseException) -> Optional[str]:
    seen: set[int] = set()
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__}: {current}".strip()
        lowered = text.casefold()
        if any(marker in lowered for marker in _GPU_ERROR_MARKERS):
            return text
        current = current.__cause__ or current.__context__
    return None


def _min_free_mib(value: Any = None) -> int:
    if value is None or str(value).strip() == "":
        value = os.environ.get(
            "SUPER_OFFLINE_GPU_MIN_FREE_MIB",
            os.environ.get("FASTWAM_GPU_MIN_FREE_MIB", DEFAULT_MIN_FREE_MIB),
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"GPU minimum free memory must be an integer MiB: {value!r}") from exc
    if parsed < 0:
        raise ValueError("GPU minimum free memory must be non-negative")
    return parsed


def _memory_info(device: Any) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": f"nvidia-smi unavailable: {exc}"}
    requested = None
    device_text = str(device or "cuda").strip().lower()
    suffix = device_text.rsplit(":", 1)[-1]
    if ":" in device_text:
        try:
            requested = int(suffix)
        except ValueError:
            pass
    rows = []
    for raw in result.stdout.splitlines():
        row = next(csv.reader([raw.strip()]), [])
        if len(row) < 3:
            continue
        try:
            rows.append((int(row[0]), float(row[1]), float(row[2])))
        except ValueError:
            continue
    if not rows:
        return {"ok": False, "error": "nvidia-smi returned no usable GPU memory data"}
    selected = next((row for row in rows if requested is None or row[0] == requested), None)
    if selected is None:
        return {
            "ok": False,
            "error": f"requested CUDA device {device!r} is not present in nvidia-smi output",
        }
    index, free_mib, total_mib = selected
    return {"ok": True, "index": index, "free_mib": free_mib, "total_mib": total_mib}


def gpu_preflight(
    device: Any,
    *,
    min_free_mib: Any = None,
    check_memory: bool = True,
    torch_module: Any = None,
) -> Dict[str, Any]:
    device_text = str(device or "").strip() or "cuda"
    if not is_cuda_device(device_text):
        return {"ok": True, "device": device_text, "skipped": True}
    memory: Dict[str, Any] = {}
    if check_memory:
        memory = _memory_info(device_text)
        if not memory.get("ok"):
            return {"ok": False, "device": device_text, **memory}
        required = _min_free_mib(min_free_mib)
        if float(memory["free_mib"]) < required:
            return {
                "ok": False,
                "device": device_text,
                "error": f"GPU {device_text} has only {memory['free_mib']:.0f} MiB free; {required} MiB is required",
                "free_mib": float(memory["free_mib"]),
                "total_mib": float(memory["total_mib"]),
                "min_free_mib": required,
            }
    if torch_module is None:
        import torch as torch_module
    try:
        available = bool(torch_module.cuda.is_available())
    except Exception as exc:
        return {"ok": False, "device": device_text, "error": f"CUDA availability check failed: {exc}"}
    if not available:
        return {"ok": False, "device": device_text, "error": "PyTorch reports CUDA is unavailable"}
    result = {"ok": True, "device": device_text}
    if check_memory:
        result.update({"free_mib": float(memory["free_mib"]), "total_mib": float(memory["total_mib"]), "min_free_mib": _min_free_mib(min_free_mib)})
    return result


def safe_cuda_cleanup(torch_module: Any = None) -> Optional[str]:
    """Best-effort allocator cleanup that never masks the original failure."""
    if GPU_FAULT_LATCH.faulted:
        return "GPU fault is latched; CUDA cleanup skipped until process restart"
    if torch_module is None:
        import torch as torch_module
    try:
        if not torch_module.cuda.is_available():
            return None
        torch_module.cuda.empty_cache()
        ipc_collect = getattr(torch_module.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            ipc_collect()
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if classify_gpu_error(exc) is not None:
            GPU_FAULT_LATCH.trip(reason)
        return reason
    return None


class GpuFaultLatch:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reason: Optional[str] = None
        self._latched_at: Optional[float] = None

    @property
    def faulted(self) -> bool:
        with self._lock:
            return self._reason is not None

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {"faulted": self._reason is not None, "fault_reason": self._reason, "faulted_at": self._latched_at}

    def trip(self, reason: str) -> bool:
        with self._lock:
            if self._reason is not None:
                return False
            self._reason = str(reason)
            self._latched_at = time.time()
            return True

    def ensure_healthy(self) -> None:
        with self._lock:
            reason = self._reason
        if reason is not None:
            raise RuntimeError(f"GPU inference is disabled after a detected GPU failure; restart the process ({reason})")


GPU_FAULT_LATCH = GpuFaultLatch()
