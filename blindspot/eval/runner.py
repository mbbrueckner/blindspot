"""Runs variants and models over test images (WP 5.1, WP 2.1).

A *variant* is a function ``variant(rgb, intrinsics) -> (depth_m, alerts)``, e.g. naive baseline, base
model or base model with ground scaling. It gets the camera intrinsics of the image (metric models such
as Metric3D need the focal length) but never the ground truth.

Two ways to run things:
    * :func:`evaluate_variant` times a variant frame by frame and scores it directly.
    * :func:`predict_models` runs registry models over samples and caches their predictions with
      performance numbers; :mod:`blindspot.eval.scoring` computes the metrics from the cache.
"""

import gc
import time
import traceback
from datetime import datetime
from functools import partial
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from blindspot import device as dev
from blindspot.core import Alert, Intrinsics, Sample
from blindspot.eval.cache import PredictionCache
from blindspot.eval.metrics import depth_metrics, latency_stats, warning_metrics
from blindspot.eval.timing import PerfRecorder, timed

Variant = Callable[[np.ndarray, Intrinsics], tuple[np.ndarray, Sequence[Alert]]]


def evaluate_variant(
    variant: Variant,
    samples: Iterable[Sample],
    warn_range_m: float = 3.0,
    max_depth: float = 10.0,
) -> dict:
    """Runs a variant over all images and computes all metrics.

    Args:
        variant: Function ``(rgb, intrinsics) -> (depth_m, alerts)``; the depth has the shape of the
            reference.
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
        pred, warns = variant(s.rgb, s.intrinsics)
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


def _cache_meta(info) -> dict:
    """What identifies a model's cached predictions (changes invalidate the cache)."""
    return {"info": info.to_dict()}


def predict_models(
    model_names: Sequence[str],
    samples: Sequence[Sample],
    cache: PredictionCache,
    device: str,
    warmup: int = 3,
    recompute: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, dict]:
    """Runs each model over all samples and caches the predictions with performance numbers.

    Frames already in the cache are skipped (unless ``recompute``). A model that fails to import, load
    or predict is reported with its reason and skipped; the run continues with the next model.

    Performance: load time, parameter count, latency per frame (after ``warmup`` untimed runs on the
    first frame, device synchronized before the timer stops) and peak memory. The numbers are stored in
    ``perf_<device>.json`` next to the predictions and reused when everything comes from the cache.

    Args:
        model_names: Registry names.
        samples: Test images (any datasets mixed).
        cache: Prediction cache.
        device: "cuda", "mps" or "cpu".
        warmup: Untimed predictions before timing.
        recompute: Ignore and overwrite cached predictions.
        log: Progress output.

    Returns:
        Dict model name -> status with ``status`` ("ok", "cached" or "failed"), ``stage`` and ``error``
        for failures, ``n_predicted``, ``n_cached``, ``perf`` and ``info``.
    """
    from blindspot.models import registry

    results = {}
    for name in model_names:
        info = registry.info(name)
        status = {"info": info.to_dict(), "status": "ok", "n_predicted": 0, "n_cached": 0, "perf": None}
        results[name] = status
        meta = _cache_meta(info)
        if recompute:
            cache.clear(name)
        elif cache.invalidate_if_stale(name, meta):
            log(f"[{name}] model info changed, cached predictions cleared")
        todo = [s for s in samples if not cache.has(name, s.frame_id)]
        status["n_cached"] = len(samples) - len(todo)
        if not todo:
            status["status"] = "cached"
            status["perf"] = cache.read_json(name, f"perf_{device}") or _any_perf(cache, name)
            log(f"[{name}] all {len(samples)} frames cached")
            continue
        model = None
        stage = "import"
        try:
            model = registry.create(name)
            stage = "load"
            _, load_s = timed(partial(model.load, device), device)
            n_params = model.n_params()
            stage = "warm-up"
            for _ in range(warmup):
                model.predict(todo[0].rgb, todo[0].intrinsics)
            stage = "predict"
            rec = PerfRecorder(device)
            for i, s in enumerate(todo):
                with rec.measure():
                    pred = model.predict(s.rgb, s.intrinsics)
                if pred.shape != s.rgb.shape[:2]:
                    raise ValueError(f"prediction shape {pred.shape} != image shape {s.rgb.shape[:2]}")
                cache.save(name, s.frame_id, pred)
                status["n_predicted"] += 1
                if (i + 1) % 25 == 0:
                    log(f"[{name}] {i + 1}/{len(todo)} frames")
            perf = {
                "device": device,
                "device_name": dev.device_name(device),
                "load_s": load_s,
                "n_params": n_params,
                "warmup": warmup,
                **rec.summary(),
                "date": datetime.now().isoformat(timespec="seconds"),
            }
            cache.write_json(name, "meta", meta)
            cache.write_json(name, f"perf_{device}", perf)
            status["perf"] = perf
            log(f"[{name}] {len(todo)} frames, {perf['mean_ms']:.1f} ms/frame, load {load_s:.1f} s")
        except Exception as e:  # noqa: BLE001 - any model failure must not stop the run
            status |= {
                "status": "failed",
                "stage": stage,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=-4),
            }
            hint = f" (install: {info.install})" if stage == "import" else ""
            log(f"[{name}] FAILED at {stage}: {type(e).__name__}: {str(e)[:300]}{hint}")
        finally:
            if model is not None:
                model.unload()
            model = None
            gc.collect()
            dev.empty_cache(device)
    return results


def _any_perf(cache: PredictionCache, name: str) -> Optional[dict]:
    """Performance numbers of any device if none exist for the current one (e.g. predicted elsewhere)."""
    for device in ("cuda", "mps", "cpu"):
        perf = cache.read_json(name, f"perf_{device}")
        if perf is not None:
            return perf
    return None
