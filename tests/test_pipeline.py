"""Runner, scoring, report and the whole pipeline with dummy models (no torch, no weights)."""

import json

import numpy as np
import pytest
from conftest import write_sanpo_session, write_stray_recording

from blindspot.core import Alert, Intrinsics, Sample
from blindspot.eval import scoring
from blindspot.eval.cache import PredictionCache
from blindspot.eval.pipeline import make_config, run
from blindspot.eval.report import Report, write_json
from blindspot.eval.runner import evaluate_variant, predict_models
from blindspot.models import registry
from blindspot.models.base import ModelInfo

DUMMIES = {
    "dummy_perfect": ("metric", "tests.dummy_models:ScaledDepth", {"factor": 1.0}),
    "dummy_far": ("metric", "tests.dummy_models:ScaledDepth", {"factor": 2.0}),
    "dummy_rel": ("relative", "tests.dummy_models:RelativeDisparity", {}),
    "dummy_broken": ("metric", "tests.dummy_models:BrokenLoad", {}),
    "dummy_shape": ("metric", "tests.dummy_models:WrongShape", {}),
    "dummy_missing": ("metric", "tests.no_such_module:Model", {}),
}


@pytest.fixture(autouse=True)
def dummy_registry():
    for name, (kind, target, kwargs) in DUMMIES.items():
        family = "Dummy rel" if kind == "relative" else "Dummy"
        registry.register(ModelInfo(name, name, family, kind, False, "test"), target, **kwargs)
    yield
    for name in DUMMIES:
        registry.unregister(name)


def make_samples(n=3, h=12, w=16):
    """Samples whose red channel encodes the reference depth (red = 20 * depth), see dummy_models."""
    out = []
    for i in range(n):
        depth = np.tile(np.linspace(1.0, 8.0, w, dtype=np.float32), (h, 1)) + 0.2 * i
        rgb = np.zeros((h, w, 3), np.uint8)
        rgb[:, :, 0] = np.round(depth * 20).astype(np.uint8)
        depth = rgb[:, :, 0].astype(np.float32) / 20  # exactly representable
        tags = {"dataset": "iphone", "env": "out", "obstacles": "pole-bike" if i else "none"}
        out.append(Sample(f"rec/{i:06d}", rgb, depth, Intrinsics(20, 20, w / 2, h / 2), tags=tags))
    return out


def test_evaluate_variant_passes_intrinsics():
    s = make_samples(1)[0]
    seen = []

    def variant(rgb, k):
        seen.append(k)
        return s.depth, [Alert("left", 2.0)]

    result = evaluate_variant(variant, [s])
    assert seen == [s.intrinsics] and result["depth"]["abs_rel"] == 0


def test_predict_models_caches_and_reports_failures(tmp_path):
    samples = make_samples()
    cache = PredictionCache(tmp_path)
    names = ["dummy_perfect", "dummy_broken", "dummy_shape", "dummy_missing", "dummy_rel"]
    status = predict_models(names, samples, cache, "cpu", warmup=1, log=lambda m: None)
    assert status["dummy_perfect"]["status"] == "ok" and status["dummy_perfect"]["n_predicted"] == 3
    assert status["dummy_broken"]["status"] == "failed" and status["dummy_broken"]["stage"] == "load"
    assert "weights not found" in status["dummy_broken"]["error"]
    assert status["dummy_shape"]["stage"] == "predict" and "shape" in status["dummy_shape"]["error"]
    assert status["dummy_missing"]["stage"] == "import"
    perf = status["dummy_perfect"]["perf"]
    assert perf["n_timed"] == 3 and perf["mean_ms"] >= 0 and perf["device"] == "cpu"
    assert cache.read_json("dummy_perfect", "perf_cpu")["n_timed"] == 3
    again = predict_models(["dummy_perfect"], samples, cache, "cpu", log=lambda m: None)
    assert again["dummy_perfect"]["status"] == "cached" and again["dummy_perfect"]["perf"]["n_timed"] == 3


def test_scoring_alignment_ranges_tags_and_warnings(tmp_path):
    samples = make_samples()
    cache = PredictionCache(tmp_path)
    predict_models(
        ["dummy_perfect", "dummy_far", "dummy_rel"], samples, cache, "cpu", warmup=0, log=lambda m: None
    )
    models = {"dummy_perfect": "metric", "dummy_far": "metric", "dummy_rel": "relative"}
    records, binned = scoring.score_frames(models, samples, cache)
    summary = {(r["model"], r["alignment"], r["range"]): r for r in scoring.aggregate(records)}
    assert summary[("dummy_perfect", "raw", "all")]["abs_rel"] == pytest.approx(0, abs=1e-3)
    assert summary[("dummy_far", "raw", "2-5")]["abs_rel"] == pytest.approx(
        1.0, abs=1e-2
    )  # 2x, below the clip
    assert summary[("dummy_far", "raw", "all")]["abs_rel"] < 1.0  # predictions above 10 m are clipped
    assert summary[("dummy_far", "median", "all")]["abs_rel"] == pytest.approx(0, abs=1e-2)
    assert summary[("dummy_rel", "affine", "0-2")]["abs_rel"] == pytest.approx(0, abs=1e-2)
    assert ("dummy_rel", "raw", "all") not in summary  # raw only for metric models
    by_tag = scoring.aggregate_by_tag(records)
    obstacles = {
        r["value"]: r["n_frames"]
        for r in by_tag
        if r["model"] == "dummy_far" and r["alignment"] == "raw" and r["tag"] == "obstacles"
    }
    assert obstacles == {"none": 1, "pole": 2, "bike": 2}  # multi-value tag counted per value
    curves = scoring.distance_curves(binned, min_pixels=1)
    far = next(c for c in curves if c["model"] == "dummy_far" and c["alignment"] == "raw")
    assert np.nanmax(far["abs_rel"]) == pytest.approx(1.0, abs=1e-2)
    from blindspot.alerts import ALERT_LOGICS, CorridorConfig

    rows = scoring.score_warnings(["dummy_perfect"], samples, cache, ALERT_LOGICS, CorridorConfig())
    assert {r["variant"] for r in rows} == {
        "dummy_perfect+corridor",
        "dummy_perfect+naive",
        "reference+corridor",
        "reference+naive",
    }
    assert all(0 <= r["alert_rate"] <= 1 and r["n_annotated"] == 0 for r in rows)


def test_report_renders_markdown_and_json(tmp_path):
    r = Report("T")
    r.h(2, "Section")
    r.p("**bold** and `code`")
    r.table(["a", "b"], [["x", float("nan")], ["y", 0.12345]])
    (tmp_path / "f.png").write_bytes(b"\x89PNG fake")
    r.figure(tmp_path.joinpath("f.png").relative_to(tmp_path), "caption")
    md = r.markdown()
    assert "| x | – |" in md and "| y | 0.123 |" in md and "![caption](f.png)" in md
    write_json(tmp_path / "r.json", {"v": float("nan"), "p": tmp_path})
    assert json.loads((tmp_path / "r.json").read_text())["v"] is None


def test_pipeline_end_to_end(tmp_path):
    write_sanpo_session(tmp_path / "sanpo", "s1", n_frames=3)
    write_stray_recording(tmp_path / "iphone", "in_corridor_artificial_glass-person_01", n_depth=4)
    write_stray_recording(tmp_path / "iphone", "not_valid", n_depth=2)
    annotations = tmp_path / "ann.json"
    annotations.write_text(json.dumps({"in_corridor_artificial_glass-person_01/000001": []}))
    cfg = make_config(
        "smoke",
        run_name="t",
        models="dummy_perfect,dummy_rel,dummy_broken",
        sanpo_root=tmp_path / "sanpo",
        iphone_root=tmp_path / "iphone",
        iphone_split=None,
        iphone_stride=1,
        iphone_scale=1,
        annotations=[annotations],
        results_root=tmp_path / "results",
        cache_root=tmp_path / "results" / "cache",
        warmup=0,
        figures=False,
    )
    out = run(cfg, log=lambda m: None)
    data = json.loads((out / "report.json").read_text())
    assert data["data"]["sanpo"]["frames"] == 3 and data["data"]["iphone"]["frames"] == 3
    assert data["data"]["iphone"]["skipped_invalid_names"] == ["not_valid"]
    assert data["data"]["annotated_frames"] == 1
    assert data["models"]["dummy_broken"]["status"] == "failed"
    assert {r["dataset"] for r in data["summary"]} == {"sanpo", "iphone"}
    md = (out / "report.md").read_text()
    assert "oracle upper bound" in md and "Failed models" in md and "dummy_broken" in md
    assert "glass" in md  # tag table of the own recording
    assert not (out / "report.html").exists()


def test_pipeline_figures(tmp_path):
    pytest.importorskip("matplotlib")
    write_stray_recording(tmp_path / "iphone", "out_street_night_car_02", n_depth=4)
    write_stray_recording(tmp_path / "iphone", "in_room_day_none_01", n_depth=4)
    cfg = make_config(
        "smoke",
        run_name="f",
        models="dummy_perfect,dummy_far,dummy_rel",
        datasets=("iphone",),
        iphone_root=tmp_path / "iphone",
        iphone_split=None,
        iphone_recordings=["out_street_night_car_02", "in_room_day_none_01"],
        iphone_stride=1,
        iphone_scale=1,
        highlight="dummy_perfect,dummy_rel,not_a_model",
        results_root=tmp_path / "results",
        cache_root=tmp_path / "results" / "cache",
        warmup=0,
    )
    out = run(cfg, log=lambda m: None)
    names = {p.name for p in (out / "figures").iterdir()}
    for stem in ("accuracy_vs_latency", "raw_vs_scaled", "near_range", "error_over_distance", "conditions"):
        assert {f"iphone_{stem}.png", f"iphone_{stem}.pdf"} <= names, stem
    md = (out / "report.md").read_text()
    assert "![iphone: Accuracy vs. speed" in md and "figures/iphone_conditions.png" in md
