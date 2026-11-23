"""Alignment, per-range metrics, cache, device helpers and the model registry."""

import numpy as np
import pytest

from blindspot import device
from blindspot.eval import alignment, metrics
from blindspot.eval.cache import PredictionCache
from blindspot.models import registry
from blindspot.models.base import disparity_to_pseudo_depth

# ---- Alignment ----


def test_resize_to_checks_aspect_ratio():
    assert alignment.resize_to(np.ones((10, 20)), (20, 40)).shape == (20, 40)
    with pytest.raises(ValueError, match="Aspect"):
        alignment.resize_to(np.ones((10, 20)), (20, 20))


def test_median_alignment_recovers_scale():
    gt = np.linspace(1, 9, 100, dtype=np.float32).reshape(10, 10)
    aligned, fit = alignment.align(gt * 0.25, gt, "median")
    assert fit["scale"] == pytest.approx(4.0) and np.allclose(aligned, gt, rtol=1e-5)


def test_affine_alignment_recovers_relative_disparity():
    gt = np.linspace(1, 9, 100, dtype=np.float32).reshape(10, 10)
    pseudo = disparity_to_pseudo_depth(3.0 / gt + 0.5)  # affine disparity, as a relative model outputs
    aligned, _ = alignment.align(pseudo, gt, "affine")
    assert np.allclose(aligned, gt, rtol=1e-4)
    median_only = alignment.align(pseudo, gt, "median")[0]
    assert metrics.depth_metrics(median_only, gt)["abs_rel"] > 0.1  # one scale cannot undo the shift


def test_clip_depth_limits_outliers():
    d = np.array([0.0, 5.0, 500.0, np.nan], np.float32)
    out = alignment.clip_depth(d, 10.0)
    assert out[0] == pytest.approx(1e-3) and out[1] == 5 and out[2] == 10 and np.isnan(out[3])


def test_alignment_raw_and_invalid():
    gt = np.full((4, 4), np.nan, np.float32)
    pred = np.ones((4, 4), np.float32)
    assert alignment.align(pred, gt, "raw")[0] is pred
    assert np.isnan(alignment.align(pred, gt, "median")[0]).all()
    with pytest.raises(ValueError):
        alignment.align(pred, gt, "best")


# ---- Metrics ----


def test_range_metrics_split_by_reference_depth():
    gt = np.array([[1.0, 3.0], [6.0, 20.0]], np.float32)
    pred = gt * np.array([[1.1, 1.0], [1.0, 1.0]], np.float32)
    m = metrics.range_metrics(pred, gt)
    assert m["0-2"]["abs_rel"] == pytest.approx(0.1) and m["0-2"]["n_pixels"] == 1
    assert m["2-5"]["abs_rel"] == 0 and m["5-10"]["n_pixels"] == 1
    assert m["all"]["n_pixels"] == 3  # 20 m is beyond max depth


def test_binned_abs_rel():
    gt = np.array([[0.2, 0.7, 0.8, np.nan]], np.float32)
    pred = np.array([[0.4, 0.7, 0.4, 1.0]], np.float32)
    sums, counts = metrics.binned_abs_rel(pred, gt, bin_m=0.5, max_depth=1.0)
    assert counts.tolist() == [1, 2] and sums == pytest.approx([1.0, 0.5])


# ---- Cache ----


def test_cache_roundtrip_float16_and_staleness(tmp_path):
    cache = PredictionCache(tmp_path)
    pred = np.array([[1.2345, 1e6]], np.float32)
    cache.save("m", "rec/000001", pred)
    assert cache.has("m", "rec/000001")
    back = cache.load("m", "rec/000001")
    assert back.dtype == np.float32 and back[0, 0] == pytest.approx(1.2345, rel=1e-3)
    assert back[0, 1] == pytest.approx(65504)  # clipped to the float16 range
    cache.write_json("m", "meta", {"v": 1, "note": "old descriptive field"})
    assert not cache.invalidate_if_stale("m", {"v": 1})  # extra stored fields are ignored
    assert cache.invalidate_if_stale("m", {"v": 2}) and not cache.has("m", "rec/000001")
    with pytest.raises(ValueError):
        cache.path("m", "../escape")


# ---- Device and registry ----


def test_device_cpu_path_works_without_gpu():
    assert device.select_device("cpu") == "cpu"
    with pytest.raises(ValueError):
        device.select_device("tpu")
    device.synchronize("cpu")
    device.reset_peak_memory("cpu")
    assert device.peak_memory_mb("cpu") is None and device.process_peak_rss_mb() > 0


def test_registry_groups_and_resolution():
    assert set(registry.GROUPS["tier1"]) == {
        "midas_s",
        "da2_s",
        "da2m_in_s",
        "da2m_out_s",
        "m3d_s",
        "yolo26n",
        "zipdepth",
    }
    assert set(registry.GROUPS["all"]) == set(registry.names())
    assert all(registry.info(n).uses_intrinsics for n in registry.GROUPS["camera"])
    assert all(registry.info(n).kind in ("metric", "relative") for n in registry.names())
    assert registry.resolve("tier1,m3d_s,depthpro")[-1] == "depthpro"
    assert len(registry.resolve("tier1,m3d_s")) == 7  # no duplicates
    with pytest.raises(KeyError, match="neither"):
        registry.resolve("nope")
