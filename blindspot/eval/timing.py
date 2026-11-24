"""Measures run time and memory of a model fairly across devices.

Rules: warm-up runs are excluded (first calls compile kernels and allocate memory), and the device is
synchronized before the timer stops, because GPU calls return before the work is done.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from blindspot import device as dev
from blindspot.eval.metrics import latency_stats


@dataclass
class PerfRecorder:
    """Collects latency and memory while a model runs over frames.

    Usage::

        rec = PerfRecorder("cuda")
        with rec.measure():
            model.predict(rgb, k)

    Attributes:
        device: "cuda", "mps" or "cpu".
        seconds: Measured per-frame run times (warm-up excluded).
        mps_peak_mb: Largest sampled MPS allocation (MPS has no peak counter).
    """

    device: str
    seconds: list[float] = field(default_factory=list)
    mps_peak_mb: float = 0.0

    def __post_init__(self) -> None:
        """Resets the device peak memory counter."""
        dev.reset_peak_memory(self.device)

    def measure(self) -> "_Measurement":
        """Context manager that times one frame (synchronized) and samples memory."""
        return _Measurement(self)

    def summary(self) -> dict:
        """Latency (mean, p95, FPS, number of frames) and peak memory.

        Returns:
            Dict with ``mean_ms``, ``p95_ms``, ``fps``, ``n_timed``, ``peak_mem_mb`` (device tensor memory:
            exact on CUDA, sampled after each frame on MPS, ``None`` on the CPU) and ``peak_rss_mb``
            (whole process, for CPU runs).
        """
        peak = dev.peak_memory_mb(self.device)
        if self.device == "mps":
            peak = self.mps_peak_mb or None
        return {
            **latency_stats(self.seconds),
            "n_timed": len(self.seconds),
            "peak_mem_mb": peak,
            "peak_rss_mb": dev.process_peak_rss_mb(),
        }


class _Measurement:
    """Context manager used by :meth:`PerfRecorder.measure`."""

    def __init__(self, recorder: PerfRecorder) -> None:
        self.recorder = recorder
        self.start = 0.0

    def __enter__(self) -> "_Measurement":
        dev.synchronize(self.recorder.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        dev.synchronize(self.recorder.device)
        if exc_type is None:
            self.recorder.seconds.append(time.perf_counter() - self.start)
        if self.recorder.device == "mps":
            current = dev.current_memory_mb("mps") or 0.0
            self.recorder.mps_peak_mb = max(self.recorder.mps_peak_mb, current)


def timed(fn, device: str) -> tuple[object, float]:
    """Calls ``fn()`` and returns (result, seconds), synchronizing the device before and after.

    Args:
        fn: Function without arguments.
        device: "cuda", "mps" or "cpu".
    """
    dev.synchronize(device)
    start = time.perf_counter()
    result = fn()
    dev.synchronize(device)
    return result, time.perf_counter() - start


def count_parameters(module: Optional[object]) -> Optional[int]:
    """Number of parameters of a torch module (``None`` if it is not a module).

    Args:
        module: Object with a ``parameters()`` method, e.g. ``torch.nn.Module``.
    """
    if module is None or not hasattr(module, "parameters"):
        return None
    return int(sum(int(np.prod(p.shape)) for p in module.parameters()))
