"""Built-in variants for the evaluation framework (WP 5.1).

A variant is a function ``rgb -> (depth_m, alerts)``, see :mod:`blindspot.evaluation`.
"""

import numpy as np

from blindspot.core import Alert

CONSTANT_DEPTH_M = 4.0


def constant_depth(rgb: np.ndarray) -> tuple[np.ndarray, list[Alert]]:
    """Predicts the same depth everywhere and never alerts; a sanity floor for every metric.

    Args:
        rgb: Image, RGB uint8, shape (H, W, 3).

    Returns:
        Depth in meters (float32, shape (H, W)) and an empty alert list.
    """
    return np.full(rgb.shape[:2], CONSTANT_DEPTH_M, np.float32), []
