"""Synthetic data for the tests: a fake Stray Scanner recording and a fake SANPO session."""

import gzip
import json

import cv2
import numpy as np
import pytest

VIDEO_W, VIDEO_H = 1920, 1440
DEPTH_W, DEPTH_H = 256, 192
FX = 1440.0


def frame_value(i: int) -> int:
    """Gray value that encodes the frame number in the fake video (robust to compression)."""
    return 20 + 40 * (i % 6)


def write_stray_recording(
    root, name="out_sidewalk_day_bollard-bike_01", n_depth=6, drop_first=True, depth_mm=2000, odometry=True
):
    """Writes a Stray Scanner recording.

    Depth frame ``d`` is constant ``depth_mm + 100 * d`` millimeters; the left quarter has confidence 1
    (must be dropped), the rest confidence 2. The video contains ``n_depth - 1`` frames if ``drop_first``
    (like the real recordings: the first frame is missing), video frame ``k`` then shows depth frame
    ``k + 1``, encoded in its gray value.
    """
    rec = root / name
    (rec / "depth").mkdir(parents=True)
    (rec / "confidence").mkdir()
    for d in range(n_depth):
        cv2.imwrite(
            str(rec / "depth" / f"{d:06d}.png"), np.full((DEPTH_H, DEPTH_W), depth_mm + 100 * d, np.uint16)
        )
        conf = np.full((DEPTH_H, DEPTH_W), 2, np.uint8)
        conf[:, : DEPTH_W // 4] = 1
        cv2.imwrite(str(rec / "confidence" / f"{d:06d}.png"), conf)
    writer = cv2.VideoWriter(str(rec / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (VIDEO_W, VIDEO_H))
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot write mp4 videos")
    for d in range(1 if drop_first else 0, n_depth):
        writer.write(np.full((VIDEO_H, VIDEO_W, 3), frame_value(d), np.uint8))
    writer.release()
    (rec / "camera_matrix.csv").write_text(f"{FX}, 0.0, 960.0\n0.0, {FX}, 720.0\n0.0, 0.0, 1.0\n")
    if odometry:
        lines = ["timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy"]
        for d in range(n_depth):
            lines.append(f"{d / 30:.4f}, {d:06d}, 0, 0, 0, 0, 0, 0, 1, {FX + d}, {FX + d}, 960.0, 720.0")
        (rec / "odometry.csv").write_text("\n".join(lines) + "\n")
    return rec


def write_sanpo_session(root, sid="sess", n_frames=2, h=6, w=8, depth_value=4.0):
    """Writes a SANPO session (chest camera) with constant depth, image at the depth resolution."""
    left = root / sid / "camera_chest" / "left"
    (left / "video_frames").mkdir(parents=True)
    (left / "depth_maps").mkdir()
    params = {"fx": 10.0, "fy": 10.0, "cx": w / 2, "cy": h / 2, "image_height": h, "image_width": w}
    desc = {
        "session_camera_location": ["camera_chest"],
        "session_camera_details": [{"left_camera_params": params}],
    }
    (root / sid / "description.json").write_text(json.dumps(desc))
    for i in range(n_frames):
        cv2.imwrite(str(left / "video_frames" / f"{i:06d}.png"), np.full((h, w, 3), 100, np.uint8))
        depth = np.full((h, w), depth_value + i, np.float16)
        with gzip.open(left / "depth_maps" / f"{i:06d}.float16.gz", "wb") as f:
            f.write(np.array([h, w], np.float16).tobytes() + depth.tobytes())
    return root / sid


@pytest.fixture
def stray_recording(tmp_path):
    return write_stray_recording(tmp_path)
