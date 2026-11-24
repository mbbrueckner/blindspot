"""Naive baseline without walking corridor (WP 3.1).

Every pixel closer than the warning range counts, no matter where it is: the floor in front of the
feet, a wall beside the path or a ceiling lamp. This is the reference the corridor logic must beat on
false alarms.
"""

import numpy as np

from blindspot.alerts.corridor import CorridorConfig, alerts_from_mask
from blindspot.core import Alert, Intrinsics


def naive_alerts(depth: np.ndarray, k: Intrinsics, cfg: CorridorConfig = CorridorConfig()) -> list[Alert]:
    """Alerts for every image third with enough pixels closer than ``cfg.warn_range_m``.

    Args:
        depth: Metric depth in meters, shape (H, W).
        k: Intrinsics (unused, same signature as :func:`~blindspot.alerts.corridor.corridor_alerts`).
        cfg: Only ``warn_range_m``, ``min_pixel_fraction`` and ``near_band_m`` are used.

    Returns:
        Alerts sorted by distance.
    """
    with np.errstate(invalid="ignore"):
        mask = np.isfinite(depth) & (depth > 0) & (depth <= cfg.warn_range_m)
    return alerts_from_mask(depth, mask, cfg.min_pixel_fraction, cfg.near_band_m)
