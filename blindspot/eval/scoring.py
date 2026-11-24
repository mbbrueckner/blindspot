"""Computes all metrics from cached predictions, so they can be recomputed without running the models.

The central data structure is a list of *frame records*, one per (model, frame, alignment, depth
range), with the frame's tags. Aggregation is a mean over frames (as in
:func:`~blindspot.eval.runner.evaluate_variant`), skipping frames without valid pixels in that range.
"""

from collections import defaultdict
from typing import Sequence

import numpy as np

from blindspot.core import Sample
from blindspot.eval.alignment import ALIGNMENTS, align, clip_depth, resize_to
from blindspot.eval.cache import PredictionCache
from blindspot.eval.metrics import DEPTH_RANGES, binned_abs_rel, range_metrics

METRICS = ("abs_rel", "rmse", "delta1")
# Tags used for grouping; "obstacles" may hold several values joined with "-".
GROUP_TAGS = ("env", "scene", "light", "obstacles")
MULTI_VALUE_TAGS = ("obstacles",)


def alignments_for(kind: str, alignments: Sequence[str] = ALIGNMENTS) -> list[str]:
    """Alignment modes that make sense for a model kind ("raw" only for metric models)."""
    return [a for a in alignments if a != "raw" or kind == "metric"]


def primary_alignment(kind: str) -> str:
    """Alignment used for panels and headline numbers: raw for metric, affine for relative models.

    Relative models predict disparity up to scale *and shift*, so a single scale factor (median) cannot
    fit them; affine alignment in disparity is the standard protocol for them.
    """
    return "raw" if kind == "metric" else "affine"


def score_frames(
    models: dict[str, str],
    samples: Sequence[Sample],
    cache: PredictionCache,
    alignments: Sequence[str] = ALIGNMENTS,
    max_depth: float = 10.0,
    bin_m: float = 0.5,
) -> tuple[list[dict], dict]:
    """Scores cached predictions of several models.

    Args:
        models: Model name -> kind ("metric" or "relative").
        samples: Test images; frames without a cached prediction are skipped.
        cache: Prediction cache.
        alignments: Alignment modes to evaluate.
        max_depth: Largest reference depth in meters, also used for alignment fits.
        bin_m: Bin width of the error-over-distance curve.

    Returns:
        (frame records, binned) where ``binned[(model, dataset, alignment)] = (sums, counts)`` of the
        per-pixel AbsRel per depth bin, pooled over frames.
    """
    records: list[dict] = []
    binned: dict[tuple, list[np.ndarray]] = {}
    for name, kind in models.items():
        for s in samples:
            if not cache.has(name, s.frame_id):
                continue
            pred = resize_to(cache.load(name, s.frame_id), s.depth.shape)
            dataset = s.tags.get("dataset", "?")
            for mode in alignments_for(kind, alignments):
                aligned, fit = align(pred, s.depth, mode, max_depth)
                aligned = clip_depth(aligned, max_depth)
                for rng, m in range_metrics(aligned, s.depth).items():
                    records.append(
                        {
                            "model": name,
                            "dataset": dataset,
                            "frame_id": s.frame_id,
                            "alignment": mode,
                            "range": rng,
                            **m,
                            "scale": fit["scale"],
                            "tags": s.tags,
                        }
                    )
                sums, counts = binned_abs_rel(aligned, s.depth, bin_m, max_depth)
                key = (name, dataset, mode)
                if key in binned:
                    binned[key][0] += sums
                    binned[key][1] += counts
                else:
                    binned[key] = [sums, counts]
    return records, {k: (v[0], v[1]) for k, v in binned.items()}


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def aggregate(
    records: Sequence[dict], keys: Sequence[str] = ("model", "dataset", "alignment", "range")
) -> list[dict]:
    """Means over frames, grouped by the given record keys.

    Args:
        records: Frame records from :func:`score_frames`.
        keys: Record keys to group by.

    Returns:
        One dict per group with the key values, the mean of each metric over frames with valid pixels,
        ``n_frames`` (frames with valid pixels) and ``n_pixels`` (sum).
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, rs in groups.items():
        valid = [r for r in rs if r["n_pixels"] > 0]
        row = dict(zip(keys, key))
        row |= {m: _mean([r[m] for r in valid]) for m in METRICS}
        row |= {"n_frames": len(valid), "n_pixels": int(sum(r["n_pixels"] for r in rs))}
        out.append(row)
    return out


def tag_values(tags: dict, tag: str) -> list[str]:
    """Values of one tag of a frame (several for multi-value tags such as obstacles)."""
    value = tags.get(tag)
    if value is None:
        return []
    return value.split("-") if tag in MULTI_VALUE_TAGS else [value]


def aggregate_by_tag(
    records: Sequence[dict], tags: Sequence[str] = GROUP_TAGS, range_name: str = "all"
) -> list[dict]:
    """Means over frames per model, dataset, alignment, tag and tag value.

    A frame with several obstacle types counts once for each type.

    Args:
        records: Frame records from :func:`score_frames`.
        tags: Tag names to group by.
        range_name: Depth range to use.

    Returns:
        Rows with ``model``, ``dataset``, ``alignment``, ``tag``, ``value``, the metric means and
        ``n_frames``.
    """
    expanded = []
    for r in records:
        if r["range"] != range_name:
            continue
        for tag in tags:
            for value in tag_values(r["tags"], tag):
                expanded.append({**r, "tag": tag, "value": value})
    return aggregate(expanded, ("model", "dataset", "alignment", "tag", "value"))


def distance_curves(binned: dict, bin_m: float = 0.5, min_pixels: int = 100) -> list[dict]:
    """Error-over-distance curves from pooled bins.

    Args:
        binned: Output of :func:`score_frames`.
        bin_m: Bin width in meters.
        min_pixels: Bins with fewer pixels are NaN (too few to be meaningful).

    Returns:
        One dict per (model, dataset, alignment) with ``bin_centers_m``, ``abs_rel`` and ``n_pixels`` lists.
    """
    out = []
    for (model, dataset, mode), (sums, counts) in binned.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(counts >= min_pixels, sums / np.maximum(counts, 1), np.nan)
        out.append(
            {
                "model": model,
                "dataset": dataset,
                "alignment": mode,
                "bin_centers_m": ((np.arange(len(sums)) + 0.5) * bin_m).tolist(),
                "abs_rel": mean.tolist(),
                "n_pixels": counts.astype(int).tolist(),
            }
        )
    return out


__all__ = [
    "DEPTH_RANGES",
    "aggregate",
    "aggregate_by_tag",
    "alignments_for",
    "distance_curves",
    "primary_alignment",
    "score_frames",
]
