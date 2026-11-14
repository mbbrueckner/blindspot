"""Tests for evaluation, SANPO loader, splits and calibration (WP 5.1, 1.2, 1.4, 1.1)."""

import gzip
import json

import cv2
import numpy as np
import pytest

from blindspot import calibration, evaluation, sanpo, splits
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
