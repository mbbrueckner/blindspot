"""Recording names, manifest and the Stray Scanner loader (WP 1.3)."""

import numpy as np
import pytest
from conftest import DEPTH_H, DEPTH_W, FX, frame_value, write_stray_recording

from blindspot.data import recordings, stray
from blindspot.data.datasets import load_iphone

# ---- Names ----


def test_parse_valid_name():
    r = recordings.parse_recording_name("out_sidewalk_day_bollard-bike_01")
    assert (r.env, r.scene, r.light, r.obstacles, r.index) == (
        "out",
        "sidewalk",
        "day",
        ("bollard", "bike"),
        1,
    )
    assert r.name == "out_sidewalk_day_bollard-bike_01"
    assert r.tags == {"env": "out", "scene": "sidewalk", "light": "day", "obstacles": "bollard-bike"}


@pytest.mark.parametrize(
    "name, fragment",
    [
        ("outside_sidewalk_day_pole_01", "in, out, trans"),
        ("out_road_day_pole_01", "corridor, room"),
        ("out_sidewalk_noon_pole_01", "day, dusk"),
        ("out_sidewalk_day_tree_01", "pole, bollard"),
        ("out_terrace_day_walls_01", "wall, plant"),
        ("out_sidewalk_day_none-pole_01", "cannot be combined"),
        ("out_sidewalk_day_pole-pole_01", "twice"),
        ("out_sidewalk_day_pole_1", "two digits"),
        ("out_sidewalk_day_01", "5 parts"),
        ("1b8a88dae2", "5 parts"),
    ],
)
def test_invalid_names_explain_allowed_values(name, fragment):
    with pytest.raises(ValueError, match=fragment):
        recordings.parse_recording_name(name)


def test_home_scene_vocabulary():
    r = recordings.parse_recording_name("out_terrace_night_plant-wall_01")
    assert r.scene == "terrace" and r.obstacles == ("plant", "wall")


def test_manifest_valid_and_invalid(tmp_path):
    p = tmp_path / "recordings.csv"
    p.write_text(
        "recording,split,camera_height_m,weather,original_name,notes\n"
        "out_sidewalk_day_bollard-bike_01,test,1.35,cloudy,1b8a88dae2,\n"
        "in_corridor_artificial_glass_01,train,,,,\n"
    )
    m = recordings.load_manifest(p)
    assert m["out_sidewalk_day_bollard-bike_01"].camera_height_m == 1.35
    assert m["in_corridor_artificial_glass_01"].split == "train"
    assert m["in_corridor_artificial_glass_01"].camera_height_m is None
    p.write_text("recording,split\nout_sidewalk_day_pole_01,validation\n")
    with pytest.raises(ValueError, match="split"):
        recordings.load_manifest(p)
    p.write_text("recording,split\nbad_name,test\n")
    with pytest.raises(ValueError, match="recordings.csv:2"):
        recordings.load_manifest(p)
    p.write_text("recording,split,color\nout_sidewalk_day_pole_01,test,red\n")
    with pytest.raises(ValueError, match="columns"):
        recordings.load_manifest(p)


def test_find_recordings_separates_invalid_names(tmp_path):
    write_stray_recording(tmp_path, "out_sidewalk_day_pole_01", n_depth=2)
    write_stray_recording(tmp_path, "1b8a88dae2", n_depth=2)
    (tmp_path / "not_a_recording").mkdir()
    assert recordings.find_recordings(tmp_path) == (["out_sidewalk_day_pole_01"], ["1b8a88dae2"])


# ---- Stray loader ----


def test_frame_offset_rules():
    assert stray.frame_offset(10, 10) == 0
    assert stray.frame_offset(9, 10) == 1
    with pytest.raises(ValueError):
        stray.frame_offset(8, 10)


@pytest.mark.parametrize("drop_first", [True, False])
def test_stray_loader_aligns_video_and_depth(tmp_path, drop_first):
    rec = write_stray_recording(tmp_path, n_depth=6, drop_first=drop_first)
    samples = stray.load_recording(rec, stride=1, scale=2)
    first = 1 if drop_first else 0  # depth frame 0 has no video frame if the first one is missing
    assert [s.frame_id for s in samples] == [f"{rec.name}/{d:06d}" for d in range(first, 6)]
    for s in samples:
        d = int(s.frame_id.split("/")[1])
        assert abs(float(s.rgb.mean()) - frame_value(d)) < 8, "image belongs to another depth frame"
        assert s.rgb.shape == (DEPTH_H * 2, DEPTH_W * 2, 3) and s.depth.shape == (DEPTH_H * 2, DEPTH_W * 2)
        assert np.nanmax(s.depth) == pytest.approx(2.0 + 0.1 * d, abs=1e-6)  # millimeters -> meters
        assert np.isnan(s.depth[:, : DEPTH_W // 2]).all()  # confidence 1 dropped (left quarter)
        k = s.intrinsics
        assert k.fx == pytest.approx((FX + d) * DEPTH_W * 2 / 1920)  # per-frame intrinsics, scaled
        assert k.cx == pytest.approx(960 * DEPTH_W * 2 / 1920)


def test_stray_loader_stride_max_frames_tags_and_fallback(tmp_path):
    rec = write_stray_recording(tmp_path, n_depth=7, odometry=False)
    samples = stray.load_recording(rec, stride=3, max_frames=1, scale=1)
    assert [s.frame_id.split("/")[1] for s in samples] == ["000003"]  # 0 has no video frame, then 3
    s = samples[0]
    assert s.tags == {
        "dataset": "iphone",
        "recording": rec.name,
        "env": "out",
        "scene": "sidewalk",
        "light": "day",
        "obstacles": "bollard-bike",
    }
    assert s.intrinsics.fx == pytest.approx(FX * DEPTH_W / 1920)  # camera_matrix.csv fallback


def test_stray_loader_rejects_invalid_name(tmp_path):
    rec = write_stray_recording(tmp_path, "96562a0c0a", n_depth=2)
    with pytest.raises(ValueError, match="5 parts"):
        stray.load_recording(rec)


def test_load_iphone_uses_manifest_split(tmp_path):
    write_stray_recording(tmp_path, "out_sidewalk_day_pole_01", n_depth=3)
    write_stray_recording(tmp_path, "in_room_artificial_none_01", n_depth=3)
    write_stray_recording(tmp_path, "abc123", n_depth=2)
    (tmp_path / "recordings.csv").write_text(
        "recording,split,weather,camera_height_m\nout_sidewalk_day_pole_01,test,rain,1.42\n"
        "in_room_artificial_none_01,train,,\n"
    )
    samples, invalid = load_iphone(tmp_path, split="test", stride=1, scale=1)
    assert invalid == ["abc123"]
    assert {s.tags["recording"] for s in samples} == {"out_sidewalk_day_pole_01"}
    assert samples[0].tags["split"] == "test" and samples[0].tags["weather"] == "rain"
    assert samples[0].tags["camera_height_m"] == "1.42"
    samples, _ = load_iphone(tmp_path, split=None, stride=1, scale=1)
    assert len({s.tags["recording"] for s in samples}) == 2
    samples, _ = load_iphone(tmp_path, split=None, stride=1, scale=1, max_recordings=1)
    assert len({s.tags["recording"] for s in samples}) == 1
    names = ["out_sidewalk_day_pole_01", "in_room_artificial_none_01"]
    samples, _ = load_iphone(tmp_path, names, split=None, stride=1, scale=1, max_recordings=1)
    assert len({s.tags["recording"] for s in samples}) == 2  # an explicit list is not cut
