"""Tests for evaluation, SANPO loader, splits and calibration (WP 5.1, 1.2, 1.4, 1.1)."""

import gzip
import importlib.util
import json
import pathlib

import cv2
import numpy as np
import pytest

from blindspot import calibration, evaluation, masks, sanpo, splits
from blindspot.core import Alert, Intrinsics, Obstacle, Sample

K = Intrinsics(100.0, 100.0, 32.0, 24.0)


def make_sample(obstacles, depth_value=2.0, w=60, h=30):
    """Image with constant reference depth and the given annotation."""
    return Sample("f", np.zeros((h, w, 3), np.uint8), np.full((h, w), depth_value, np.float32), K, obstacles)


# ---- Depth metrics ----


def test_depth_metrics_perfect_and_scaled():
    gt = np.full((4, 4), 2.0, np.float32)
    perfect = evaluation.depth_metrics(gt, gt)
    assert perfect["abs_rel"] == 0 and perfect["rmse"] == 0 and perfect["delta1"] == 1
    scaled = evaluation.depth_metrics(gt * 1.5, gt)
    assert scaled["abs_rel"] == pytest.approx(0.5) and scaled["delta1"] == 0  # 1.5 > 1.25


def test_depth_metrics_ignores_invalid_and_far_pixels():
    gt = np.array([[2.0, np.nan], [0.0, 50.0]], np.float32)
    pred = np.full((2, 2), 2.0, np.float32)
    m = evaluation.depth_metrics(pred, gt, max_depth=10.0)
    assert m["n_pixels"] == 1 and m["abs_rel"] == 0


def test_depth_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError):
        evaluation.depth_metrics(np.ones((2, 2)), np.ones((3, 3)))


# ---- Alert metrics ----


def test_direction_of_box():
    assert evaluation.direction_of_box((0, 0, 10, 10), 60) == "left"
    assert evaluation.direction_of_box((25, 0, 35, 10), 60) == "center"
    assert evaluation.direction_of_box((50, 0, 60, 10), 60) == "right"


def test_warning_metrics_all_cases():
    near_left = [Obstacle((0, 0, 10, 10), "pole", True)]
    samples = [
        make_sample(near_left),  # TP, direction correct
        make_sample(near_left),  # FN
        make_sample(near_left),  # TP, direction wrong
        make_sample([]),  # FP
        make_sample([]),  # TN
        make_sample(None),  # not annotated, not counted
        make_sample([Obstacle((0, 0, 10, 10), "wall", False)]),  # not in path: negative
    ]
    warns = [
        [Alert("left", 2.5)],
        [],
        [Alert("right", 1.0)],
        [Alert("center", 1.0)],
        [],
        [Alert("left", 1.0)],
        [],
    ]
    m = evaluation.warning_metrics(samples, warns)
    assert m["n_positive"] == 3 and m["n_negative"] == 3
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["false_alarm_rate"] == pytest.approx(1 / 3)
    assert m["direction_accuracy"] == pytest.approx(1 / 2)
    assert m["distance_mae_m"] == pytest.approx((0.5 + 1.0) / 2)


def test_warning_range_excludes_far_obstacles():
    far = make_sample([Obstacle((0, 0, 10, 10), "pole", True)], depth_value=8.0)
    m = evaluation.warning_metrics([far], [[]], warn_range_m=3.0)
    assert m["n_positive"] == 0 and m["n_negative"] == 1


def test_latency_stats():
    s = evaluation.latency_stats([0.1, 0.1, 0.1, 0.5])
    assert s["mean_ms"] == pytest.approx(200) and s["fps"] == pytest.approx(5)
    assert np.isnan(evaluation.latency_stats([])["mean_ms"])


def test_evaluate_variant_end_to_end():
    sample = make_sample([Obstacle((0, 0, 10, 10), "pole", True)])
    result = evaluation.evaluate_variant(lambda rgb: (sample.depth, [Alert("left", 2.0)]), [sample])
    assert result["n_frames"] == 1
    assert result["depth"]["abs_rel"] == 0
    assert result["warnings"]["recall"] == 1 and result["warnings"]["direction_accuracy"] == 1
    assert result["latency"]["mean_ms"] >= 0


def test_load_annotations(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"x": [{"box": [1, 2, 3, 4], "label": "pole", "in_path": True}], "y": []}))
    ann = evaluation.load_annotations(str(p))
    assert ann["x"] == [Obstacle((1, 2, 3, 4), "pole", True)] and ann["y"] == []


# ---- SANPO loader ----


def write_session(root, h=6, w=8, depth_h=3, depth_w=4):
    """Creates an artificial session in the SANPO layout (depth half the image size, with header)."""
    sid = root / "sess"
    left = sid / "camera_chest" / "left"
    (left / "video_frames").mkdir(parents=True)
    (left / "depth_maps").mkdir()
    desc = {
        "session_camera_location": ["camera_chest", "camera_head"],
        "session_camera_details": [
            {
                "left_camera_params": {
                    "fx": 10,
                    "fy": 11,
                    "cx": 4,
                    "cy": 3,
                    "image_height": h,
                    "image_width": w,
                }
            },
            {
                "left_camera_params": {
                    "fx": 99,
                    "fy": 99,
                    "cx": 1,
                    "cy": 1,
                    "image_height": h,
                    "image_width": w,
                }
            },
        ],
    }
    (sid / "description.json").write_text(json.dumps(desc))
    cv2.imwrite(str(left / "video_frames" / "000003.png"), np.full((h, w, 3), (10, 20, 30), np.uint8))  # BGR
    depth = np.full((depth_h, depth_w), 2.5, np.float16)
    depth[0, 0] = 0
    with gzip.open(left / "depth_maps" / "000003.float16.gz", "wb") as f:
        f.write(np.array([depth_h, depth_w], np.float16).tobytes() + depth.tobytes())
    return sid


def test_sanpo_load_sample(tmp_path):
    sid = write_session(tmp_path)
    assert sanpo.frame_indices(sid) == [3]
    s = sanpo.load_sample(sid, 3)
    assert s.rgb.shape == (3, 4, 3) and s.depth.shape == (3, 4)  # image resized to depth resolution
    assert tuple(s.rgb[0, 0]) == (30, 20, 10)  # RGB instead of BGR
    assert s.depth.dtype == np.float32 and s.depth[1, 1] == 2.5 and np.isnan(s.depth[0, 0])
    assert s.intrinsics == Intrinsics(5, 5.5, 2, 1.5)  # chest not head camera, half resolution
    assert s.frame_id == "sess/camera_chest/000003"


def test_sanpo_depth_header_mismatch_raises(tmp_path):
    path = tmp_path / "bad.float16.gz"
    with gzip.open(path, "wb") as f:
        f.write(np.array([3, 4, 1, 1], np.float16).tobytes())  # header says 3x4, but only 2 pixels
    with pytest.raises(ValueError):
        sanpo.read_depth(path)


def test_sanpo_missing_camera_raises():
    with pytest.raises(ValueError):
        sanpo.read_intrinsics({"session_camera_location": ["camera_head"], "session_camera_details": []})


# ---- Splits ----


def test_splits_are_disjoint_deterministic_and_complete():
    train = [f"s{i}" for i in range(20)]
    a = splits.make_splits(train, ["t1", "t2"], 0.15, seed=1)
    assert a == splits.make_splits(train, ["t1", "t2"], 0.15, seed=1)
    assert len(a["val"]) == 3 and set(a["train"]) | set(a["val"]) == set(train)
    assert not set(a["train"]) & set(a["val"]) and a["test"] == ["t1", "t2"]


def test_splits_reject_overlap_and_write(tmp_path):
    with pytest.raises(ValueError):
        splits.make_splits(["a", "b"], ["b"])
    splits.write_splits({"train": ["a"], "val": ["b"], "test": ["c"]}, tmp_path, "x")
    assert (tmp_path / "x_val.txt").read_text() == "b\n"


# ---- Calibration ----


def test_calibration_recovers_intrinsics():
    pattern, square = (9, 6), 0.025
    true = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    obj = np.zeros((54, 3), np.float32)
    obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * square
    rng = np.random.default_rng(0)
    corners = []
    for _ in range(8):
        rvec = rng.uniform(-0.4, 0.4, 3)
        tvec = np.array([-0.1, -0.06, 0.5]) + rng.uniform(-0.05, 0.05, 3)
        pts, _ = cv2.projectPoints(obj, rvec, tvec, true, None)
        corners.append(pts.reshape(-1, 2))
    k, rms = calibration.calibrate_from_corners(corners, pattern, square, (640, 480))
    assert rms < 0.5 and k.fx == pytest.approx(800, rel=0.02) and k.cx == pytest.approx(320, abs=5)


def test_calibration_needs_enough_images():
    with pytest.raises(ValueError):
        calibration.calibrate_from_corners([np.zeros((54, 2))] * 2, (9, 6), 0.025, (640, 480))


def test_scale_and_roundtrip_calibration(tmp_path):
    assert calibration.scale_intrinsics(K, 0.5, 0.5) == Intrinsics(50, 50, 16, 12)
    path = tmp_path / "k.json"
    calibration.save_calibration(K, (64, 48), 1.35, "test", path)
    assert calibration.load_calibration(path) == (K, (64, 48), 1.35)


# ---- Annotation, split script and evaluation runner (M1 glue) ----

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"


def run_script(name, *args):
    """Imports ``scripts/<name>.py`` and calls its ``main`` with the given arguments."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main([str(a) for a in args])


def test_apply_annotations_matches_by_frame_id_and_validates_boxes():
    a, b = make_sample(None), make_sample(None)
    a.frame_id, b.frame_id = "a", "b"
    ob = Obstacle((1, 1, 10, 10), "pole", True)
    assert evaluation.apply_annotations([a, b], {"a": [ob], "x": []}) == 1
    assert a.obstacles == [ob] and b.obstacles is None  # unmatched stays "not annotated"
    for bad in [(1, 1, 61, 10), (5, 5, 5, 9), (-1, 0, 4, 4)]:  # sample is 60x30
        with pytest.raises(ValueError):
            evaluation.apply_annotations([a], {"a": [Obstacle(bad, "pole", True)]})


def test_load_samples_and_missing_session(tmp_path):
    write_session(tmp_path)
    samples = sanpo.load_samples(tmp_path, ["sess"])
    assert [s.frame_id for s in samples] == ["sess/camera_chest/000003"]
    with pytest.raises(FileNotFoundError):
        sanpo.load_samples(tmp_path, ["nope"])


def test_make_splits_script(tmp_path):
    folder = tmp_path / "data" / "real"
    folder.mkdir(parents=True)
    (folder / "train_sessions.txt").write_text("".join(f"s{i}\n" for i in range(10)))
    (folder / "test_sessions.txt").write_text("t1\nt2\n")
    out = tmp_path / "splits"
    run_script("make_splits", "--kind", "real", "--data", tmp_path / "data", "--out", out)
    assert (out / "sanpo_real_test.txt").read_text() == "t1\nt2\n"
    train = (out / "sanpo_real_train.txt").read_text().split()
    val = (out / "sanpo_real_val.txt").read_text().split()
    assert len(train) + len(val) == 10 and not set(train) & set(val)
    with pytest.raises(SystemExit):  # synthetic was never downloaded
        run_script("make_splits", "--kind", "synthetic", "--data", tmp_path / "data", "--out", out)


def test_evaluate_script_end_to_end(tmp_path):
    write_session(tmp_path)
    split = tmp_path / "test.txt"
    split.write_text("sess\n")
    ann = tmp_path / "ann.json"  # depth is 2.5 m except one invalid pixel, so the obstacle is in range
    ann.write_text(
        json.dumps({"sess/camera_chest/000003": [{"box": [1, 1, 4, 3], "label": "pole", "in_path": True}]})
    )
    out = tmp_path / "res.json"
    args = ["--root", tmp_path, "--split", split, "--out", out, "--annotations", ann]
    run_script("evaluate", *args, "--variant", "blindspot.variants:constant_depth")
    res = json.loads(out.read_text())
    assert res["n_frames"] == 1 and res["n_annotated"] == 1
    assert (
        res["warnings"]["n_positive"] == 1 and res["warnings"]["recall"] == 0.0
    )  # sanity floor never alerts
    assert res["depth"]["abs_rel"] == pytest.approx((4.0 - 2.5) / 2.5)
    with pytest.raises(ValueError):
        run_script("evaluate", *args, "--variant", "no_colon")


# ---- Masks, annotation script and mask-aware download plan (WP 1.4) ----


def make_mask(h=40, w=60):
    """Mask image (RGB) with one pole, two trees (second instance ID via the low channel), ground, speck."""
    m = np.zeros((h, w, 3), np.uint8)
    m[:, :, 0] = 3  # sidewalk everywhere, never an obstacle
    m[0:30, 28:32] = (24, 0, 1)  # pole in the center
    m[0:20, 2:10] = (28, 0, 1)  # tree on the left
    m[0:20, 50:58] = (28, 0, 2)  # second tree instance on the right
    m[35, 5] = (24, 0, 2)  # single pixel, too small
    return m


def test_obstacles_from_mask_instances_labels_and_in_path():
    obs = masks.obstacles_from_mask(make_mask(), (40, 60), min_area_px=10)
    got = sorted((o.label, o.box, o.in_path) for o in obs)
    assert got == [
        ("pole", (28, 0, 32, 30), True),
        ("tree", (2, 0, 10, 20), False),  # left of the central third (20 to 40)
        ("tree", (50, 0, 58, 20), False),
    ]


def test_obstacles_from_mask_resizes_to_target():
    obs = masks.obstacles_from_mask(make_mask(), (20, 30), min_area_px=1)
    assert any(o.label == "pole" and o.box == (14, 0, 16, 15) for o in obs)


def test_in_path_overlaps_central_third():
    assert masks.in_path((10, 0, 21, 5), 60) and masks.in_path((39, 0, 50, 5), 60)
    assert not masks.in_path((0, 0, 20, 5), 60) and not masks.in_path((40, 0, 60, 5), 60)


def test_load_obstacles_missing_mask_is_none(tmp_path):
    sid = write_session(tmp_path)
    assert masks.load_obstacles(sid, 3, (3, 4)) is None
    d = sid / "camera_chest" / "left" / "segmentation_masks"
    d.mkdir()
    cv2.imwrite(str(d / "000003.png"), cv2.cvtColor(make_mask(), cv2.COLOR_RGB2BGR))
    assert len(masks.load_obstacles(sid, 3, (40, 60), min_area_px=10)) == 3


def test_make_annotations_script_roundtrip(tmp_path):
    sid = write_session(tmp_path)
    d = sid / "camera_chest" / "left" / "segmentation_masks"
    d.mkdir()
    pole = np.zeros((6, 8, 3), np.uint8)
    pole[:, 3:5] = (24, 0, 1)  # full-height pole in the middle
    cv2.imwrite(str(d / "000003.png"), cv2.cvtColor(pole, cv2.COLOR_RGB2BGR))
    split = tmp_path / "s.txt"
    split.write_text("sess\n")
    out = tmp_path / "ann.json"
    run_script("make_annotations", "--root", tmp_path, "--split", split, "--out", out, "--min-area", 1)
    ann = evaluation.load_annotations(str(out))
    assert ann == {
        "sess/camera_chest/000003": [Obstacle((2, 0, 3, 3), "pole", True)]
    }  # nearest neighbor keeps one of two columns


def test_plan_session_with_masks_keeps_only_human_frames(monkeypatch):
    mod = importlib.util.module_from_spec(
        spec := importlib.util.spec_from_file_location("dl", SCRIPTS / "download_sanpo.py")
    )
    spec.loader.exec_module(mod)
    left = "sanpo-real/s1/camera_chest/left"
    monkeypatch.setattr(
        mod,
        "list_objects",
        lambda prefix: (
            [(f"{left}/video_frames/{i:06d}.png", 1) for i in range(0, 13, 6)]
            + [(f"{left}/video_frames/000003.png", 1)]
        ),
    )
    kinds = {
        "0": "HUMAN_ANNOTATED",
        "3": "MACHINE_ANNOTATED",
        "6": "HUMAN_ANNOTATED",
        "12": "MACHINE_ANNOTATED",
    }
    monkeypatch.setattr(mod, "_get", lambda url: json.dumps(kinds).encode())
    names = [n for n, _ in mod.plan_session("real", "s1", 1, 10, masks=True)]
    assert f"{left}/segmentation_masks/000006.png" in names and f"{left}/video_frames/000003.png" not in names
    assert (
        f"{left}/video_frames/000012.png" not in names
        and f"{left}/frame_segmentation_annotation_type.json" in names
    )

    def not_found(url):
        raise mod.urllib.error.HTTPError(url, 404, "nf", None, None)

    monkeypatch.setattr(mod, "_get", not_found)
    assert mod.plan_session("real", "s1", 1, 10, masks=True) == []  # session without masks
