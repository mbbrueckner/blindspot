"""Aligns predicted depth to the reference before scoring.

Three modes:
    * ``raw``: the prediction as it is, in meters. Only meaningful for metric models.
    * ``median``: one scale factor per image, ``median(gt) / median(pred)`` over the valid pixels.
    * ``affine``: scale and shift in disparity (1/depth), fitted by least squares per image. This is the
      standard protocol for relative models such as MiDaS, which predict affine-invariant disparity.

``median`` and ``affine`` use the ground truth and are therefore an *oracle upper bound*, not something
the system can do at run time. They make relative and metric models comparable; the ``raw`` numbers
are what a metric model really delivers. WP 2.2 replaces the oracle by ground-plane scaling.
"""

import cv2
import numpy as np

ALIGNMENTS = ("raw", "median", "affine")

MIN_DEPTH = 1e-3  # meters; lower end of the clip before scoring
_MIN_DISPARITY = 1e-3  # 1 / 1000 m; keeps aligned depth finite where the fitted disparity is <= 0


def resize_to(pred: np.ndarray, shape: tuple[int, int], max_aspect_error: float = 0.02) -> np.ndarray:
    """Resizes a prediction to the reference resolution (bilinear).

    Args:
        pred: Prediction, shape (h, w).
        shape: Target (H, W).
        max_aspect_error: Allowed relative difference of the aspect ratios.

    Returns:
        float32 array of shape ``shape`` (the input itself if the shape already matches).

    Raises:
        ValueError: If the aspect ratios differ, because resizing would distort the image.
    """
    if pred.shape == tuple(shape):
        return pred.astype(np.float32, copy=False)
    h, w = pred.shape
    if abs((w / h) / (shape[1] / shape[0]) - 1) > max_aspect_error:
        raise ValueError(f"Aspect ratio of prediction {pred.shape} does not match reference {tuple(shape)}")
    return cv2.resize(pred.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def valid_mask(pred: np.ndarray, gt: np.ndarray, max_depth: float) -> np.ndarray:
    """Pixels with finite positive reference up to ``max_depth`` and a finite positive prediction."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(gt) & (gt > 0) & (gt <= max_depth) & np.isfinite(pred) & (pred > 0)


def fit_median(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """Scale factor ``median(gt) / median(pred)`` over ``mask`` (NaN if the mask is empty)."""
    if not mask.any():
        return float("nan")
    return float(np.median(gt[mask]) / np.median(pred[mask]))


def fit_affine_disparity(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Least-squares fit ``1/gt ~ s * (1/pred) + t`` over ``mask``.

    Returns:
        (s, t); NaN if fewer than two pixels or the disparity is constant.
    """
    if mask.sum() < 2:
        return float("nan"), float("nan")
    x = 1.0 / pred[mask].astype(np.float64)
    y = 1.0 / gt[mask].astype(np.float64)
    a = np.stack([x, np.ones_like(x)], axis=1)
    (s, t), *_ = np.linalg.lstsq(a, y, rcond=None)
    if not np.isfinite(s) or x.std() == 0:
        return float("nan"), float("nan")
    return float(s), float(t)


def align(pred: np.ndarray, gt: np.ndarray, mode: str, max_depth: float = 10.0) -> tuple[np.ndarray, dict]:
    """Aligns a prediction to the reference.

    Args:
        pred: Predicted depth (meters, or pseudo-depth for relative models), resized to ``gt.shape``.
        gt: Reference depth in meters, NaN where invalid.
        mode: "raw", "median" or "affine", see module description.
        max_depth: Only reference pixels up to this depth are used for the fit.

    Returns:
        (aligned depth in meters, fit parameters ``{"scale": ..., "shift": ...}``). If the fit fails
        (no valid pixels) the aligned depth is all NaN.

    Raises:
        ValueError: If ``mode`` is unknown or the shapes differ.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, gt {gt.shape}; resize first (resize_to)")
    pred = pred.astype(np.float32, copy=False)
    if mode == "raw":
        return pred, {"scale": 1.0, "shift": 0.0}
    mask = valid_mask(pred, gt, max_depth)
    if mode == "median":
        s = fit_median(pred, gt, mask)
        return (pred * s).astype(np.float32), {"scale": s, "shift": 0.0}
    if mode == "affine":
        s, t = fit_affine_disparity(pred, gt, mask)
        with np.errstate(divide="ignore", invalid="ignore"):
            disparity = s / pred + t
        disparity = np.where(np.isfinite(disparity), np.maximum(disparity, _MIN_DISPARITY), np.nan)
        return (1.0 / disparity).astype(np.float32), {"scale": s, "shift": t}
    raise ValueError(f"Unknown alignment {mode!r}, allowed: {', '.join(ALIGNMENTS)}")


def clip_depth(depth: np.ndarray, max_depth: float) -> np.ndarray:
    """Clips aligned depth to [MIN_DEPTH, max_depth] before scoring (standard NYU/KITTI protocol).

    Without the clip, a few pixels predicted at hundreds of meters (e.g. Depth Pro in front of a wall)
    dominate the AbsRel of a frame. NaN stays NaN.

    Args:
        depth: Aligned depth in meters.
        max_depth: Upper end, the largest evaluated reference depth.
    """
    return np.clip(depth, MIN_DEPTH, max_depth)
