"""Runs a variant over test images and scores it with the metrics of :mod:`blindspot.eval.metrics` (WP 5.1).

A *variant* is a function ``variant(rgb) -> (depth_m, alerts)``, e.g. naive baseline, base model or
base model with ground scaling. The framework times each image and computes depth error, recall,
false alarms, direction accuracy and latency.
"""

import time
from typing import Callable, Iterable, Sequence

import numpy as np

from blindspot.core import Alert, Sample
from blindspot.eval.metrics import depth_metrics, latency_stats, warning_metrics

Variant = Callable[[np.ndarray], tuple[np.ndarray, Sequence[Alert]]]


def evaluate_variant(
    variant: Variant,
    samples: Iterable[Sample],
    warn_range_m: float = 3.0,
    max_depth: float = 10.0,
) -> dict:
    """Runs a variant over all images and computes all metrics.

    Args:
        variant: Function ``rgb -> (depth_m, alerts)``; the depth has the shape of the reference.
        samples: Test images.
        warn_range_m: Warning range in meters, see :func:`~blindspot.eval.metrics.warning_metrics`.
        max_depth: Largest reference depth in meters that counts, see
            :func:`~blindspot.eval.metrics.depth_metrics`.

    Returns:
        Dict with ``depth`` (mean over images), ``warnings``, ``latency`` and ``n_frames``.
    """
    samples = list(samples)
    per_frame, all_warnings, times = [], [], []
    for s in samples:
        start = time.perf_counter()
        pred, warns = variant(s.rgb)
        times.append(time.perf_counter() - start)
        per_frame.append(depth_metrics(pred, s.depth, max_depth))
        all_warnings.append(list(warns))

    depth = {}
    for key in ("abs_rel", "rmse", "delta1"):
        vals = [m[key] for m in per_frame if m["n_pixels"] > 0]
        depth[key] = float(np.mean(vals)) if vals else float("nan")
    return {
        "n_frames": len(samples),
        "depth": depth,
        "warnings": warning_metrics(samples, all_warnings, warn_range_m),
        "latency": latency_stats(times),
    }
