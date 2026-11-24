"""Device handling: the only module with CUDA- or MPS-specific code.

Everything else asks this module for the device, for synchronization before stopping a timer and for
peak memory. torch is imported lazily, so the module also works in environments without torch
(``select_device`` then returns "cpu").

On Apple Silicon, set ``PYTORCH_ENABLE_MPS_FALLBACK=1`` so operators without an MPS kernel fall back
to the CPU instead of raising an error.
"""

import os
import resource
import sys
from typing import Optional

DEVICES = ("auto", "cuda", "mps", "cpu")


def _torch():
    """Returns the torch module, or ``None`` if torch is not installed."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def select_device(preference: str = "auto") -> str:
    """Chooses the compute device.

    Args:
        preference: "auto" (CUDA, then MPS, then CPU), or an explicit "cuda", "mps" or "cpu".

    Returns:
        "cuda", "mps" or "cpu".

    Raises:
        ValueError: If ``preference`` is unknown or the requested device is not available.
    """
    if preference not in DEVICES:
        raise ValueError(f"Unknown device {preference!r}, allowed: {', '.join(DEVICES)}")
    torch = _torch()
    available = {"cpu"}
    if torch is not None:
        if torch.cuda.is_available():
            available.add("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            available.add("mps")
    if preference == "auto":
        return next(d for d in ("cuda", "mps", "cpu") if d in available)
    if preference not in available:
        raise ValueError(f"Device {preference!r} not available (available: {sorted(available)})")
    return preference


def synchronize(device: str) -> None:
    """Waits until all queued work on the device is done (no-op on the CPU).

    GPU calls are asynchronous; without synchronizing, a timer would only measure the kernel launch.

    Args:
        device: "cuda", "mps" or "cpu".
    """
    torch = _torch()
    if torch is None:
        return
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def empty_cache(device: str) -> None:
    """Releases cached GPU memory, e.g. between two models.

    Args:
        device: "cuda", "mps" or "cpu".
    """
    torch = _torch()
    if torch is None:
        return
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def reset_peak_memory(device: str) -> None:
    """Resets the peak memory counter (CUDA only; MPS and CPU have no resettable counter).

    Args:
        device: "cuda", "mps" or "cpu".
    """
    torch = _torch()
    if torch is not None and device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def current_memory_mb(device: str) -> Optional[float]:
    """Memory currently allocated by tensors on the device, in MiB.

    Args:
        device: "cuda", "mps" or "cpu".

    Returns:
        MiB, or ``None`` on the CPU (see :func:`process_peak_rss_mb` instead).
    """
    torch = _torch()
    if torch is None:
        return None
    if device == "cuda":
        return torch.cuda.memory_allocated() / 2**20
    if device == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    return None


def peak_memory_mb(device: str) -> Optional[float]:
    """Peak tensor memory on the device since the last :func:`reset_peak_memory`, in MiB.

    Only CUDA keeps an exact peak counter. For MPS the caller has to sample
    :func:`current_memory_mb` after each step (see :class:`blindspot.eval.timing.PerfRecorder`).

    Args:
        device: "cuda", "mps" or "cpu".

    Returns:
        MiB on CUDA, otherwise ``None``.
    """
    torch = _torch()
    if torch is not None and device == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    return None


def process_peak_rss_mb() -> float:
    """Peak resident memory of the whole Python process in MiB (includes data and libraries)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # bytes on macOS, KiB on Linux


def device_name(device: str) -> str:
    """Human-readable name of the device for reports, e.g. "NVIDIA GeForce RTX 5070 Ti".

    Args:
        device: "cuda", "mps" or "cpu".
    """
    torch = _torch()
    if device == "cuda" and torch is not None:
        return torch.cuda.get_device_name()
    if device == "mps":
        return "Apple Silicon (MPS)"
    return f"CPU ({os.cpu_count()} threads)"
