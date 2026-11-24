"""Metrics of the evaluation framework (WP 5.1): depth error, alert quality and latency.

Alert metrics work on image level:
    * An image is *positive* if it contains an ``in_path`` obstacle within the warning range.
    * A variant reports an image if it emits at least one alert.
    * Recall = reported positive images / all positive images.
    * False alarm rate = reported negative images / all negative images.
    * Direction: does the nearest alert point the same way as the nearest obstacle?
Multi-frame events (one obstacle over several frames) are deliberately not part of M1.
"""

from typing import Optional, Sequence

import numpy as np

from blindspot.core import DIRECTIONS, Alert, Sample


def depth_metrics(pred: np.ndarray, gt: np.ndarray, max_depth: float = 10.0) -> dict:
    """Computes depth errors over all valid reference pixels.

    Args:
        pred: Predicted depth in meters, shape (H, W).
        gt: Reference depth in meters, shape (H, W); NaN or <= 0 is invalid.
        max_depth: Reference pixels beyond this distance (meters) are ignored.

    Returns:
        Dict with ``abs_rel``, ``rmse`` (meters), ``delta1`` (share with max(p/g, g/p) < 1.25)
        and ``n_pixels``. The values are NaN if there are no valid pixels.

    Raises:
        ValueError: If ``pred`` and ``gt`` have different shapes.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, gt {gt.shape}")
    valid = np.isfinite(gt) & (gt > 0) & (gt <= max_depth) & np.isfinite(pred) & (pred > 0)
    n = int(valid.sum())
    if n == 0:
        return {
            "abs_rel": float("nan"),
            "rmse": float("nan"),
            "delta1": float("nan"),
            "n_pixels": 0,
        }
    p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)
    ratio = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "n_pixels": n,
    }


def direction_of_box(box: tuple[int, int, int, int], width: int) -> str:
    """Assigns the center of a rectangle to one third of the image.

    Args:
        box: (x0, y0, x1, y1) in pixels.
        width: Image width in pixels.

    Returns:
        "left", "center" or "right".
    """
    center_x = (box[0] + box[2]) / 2
    return DIRECTIONS[min(int(3 * center_x / width), 2)]


def obstacle_distance(depth: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Distance of an obstacle in meters: median of the valid reference depth inside the rectangle.

    Args:
        depth: Reference depth in meters, shape (H, W).
        box: (x0, y0, x1, y1) in pixels.

    Returns:
        Distance in meters, NaN if the rectangle contains no valid depth.
    """
    x0, y0, x1, y1 = box
    patch = depth[y0:y1, x0:x1]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(patch)) if patch.size else float("nan")


def _frame_truth(sample: Sample, warn_range_m: float) -> Optional[list[tuple[float, str]]]:
    """Returns (distance, direction) of all alert targets in an image, ``None`` if not annotated."""
    if sample.obstacles is None:
        return None
    width = sample.rgb.shape[1]
    targets = []
    for ob in sample.obstacles:
        if not ob.in_path:
            continue
        dist = obstacle_distance(sample.depth, ob.box)
        if np.isfinite(dist) and dist <= warn_range_m:
            targets.append((dist, direction_of_box(ob.box, width)))
    return targets


def warning_metrics(
    samples: Sequence[Sample],
    warnings: Sequence[Sequence[Alert]],
    warn_range_m: float = 3.0,
) -> dict:
    """Scores alerts against the obstacle annotation (image level, see module description).

    Args:
        samples: Images with annotation; unannotated ones (``obstacles is None``) are skipped.
        warnings: Alerts of the variant per image, same order as ``samples``.
        warn_range_m: Obstacles up to this distance (meters) should trigger an alert.

    Returns:
        Dict with ``recall``, ``false_alarm_rate``, ``direction_accuracy``, ``distance_mae_m`` and
        the counts ``n_positive`` and ``n_negative``. Values that cannot be computed are NaN.
    """
    tp = fn = fp = tn = 0
    dir_hits = dir_total = 0
    dist_errors = []
    for sample, warns in zip(samples, warnings):
        truth = _frame_truth(sample, warn_range_m)
        if truth is None:
            continue
        if truth:
            if not warns:
                fn += 1
                continue
            tp += 1
            true_dist, true_dir = min(truth)
            nearest = min(warns, key=lambda w: w.distance_m)
            dir_total += 1
            dir_hits += nearest.direction == true_dir
            dist_errors.append(abs(nearest.distance_m - true_dist))
        elif warns:
            fp += 1
        else:
            tn += 1

    def ratio(a: int, b: int) -> float:
        return a / b if b else float("nan")

    return {
        "recall": ratio(tp, tp + fn),
        "false_alarm_rate": ratio(fp, fp + tn),
        "direction_accuracy": ratio(dir_hits, dir_total),
        "distance_mae_m": float(np.mean(dist_errors)) if dist_errors else float("nan"),
        "n_positive": tp + fn,
        "n_negative": fp + tn,
    }


def latency_stats(seconds: Sequence[float]) -> dict:
    """Summarizes per-image run times.

    Args:
        seconds: Run time per image in seconds.

    Returns:
        Dict with ``mean_ms``, ``p95_ms`` and ``fps`` (derived from the mean).
    """
    if len(seconds) == 0:
        return {"mean_ms": float("nan"), "p95_ms": float("nan"), "fps": float("nan")}
    arr = np.asarray(seconds, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean() * 1000),
        "p95_ms": float(np.percentile(arr, 95) * 1000),
        "fps": float(1 / arr.mean()),
    }
