"""Compatibility module: the evaluation framework now lives in :mod:`blindspot.eval`.

Kept so that existing imports (``from blindspot.evaluation import ...``) keep working. New code
should import from :mod:`blindspot.eval.metrics`, :mod:`blindspot.eval.runner` and
:mod:`blindspot.data.annotations`.
"""

from blindspot.data.annotations import apply_annotations, load_annotations
from blindspot.eval.metrics import (
    _frame_truth,
    depth_metrics,
    direction_of_box,
    latency_stats,
    obstacle_distance,
    warning_metrics,
)
from blindspot.eval.runner import Variant, evaluate_variant

__all__ = [
    "Variant",
    "_frame_truth",
    "apply_annotations",
    "depth_metrics",
    "direction_of_box",
    "evaluate_variant",
    "latency_stats",
    "load_annotations",
    "obstacle_distance",
    "warning_metrics",
]
