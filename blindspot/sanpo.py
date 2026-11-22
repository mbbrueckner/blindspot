"""SANPO loader (WP 1.2): reads image, depth in meters and camera parameters of a session.

Expected directory layout (matches the public bucket, see ``scripts/download_sanpo.py``)::

    <root>/<session_id>/description.json
    <root>/<session_id>/<camera>/left/video_frames/000000.png
    <root>/<session_id>/<camera>/left/depth_maps/000000.float16.gz

For us ``camera`` is ``camera_chest`` (chest camera). Depth is float16 in meters, gzip-compressed.
The first two float16 values are a header (height, width), followed by the pixels row by row.
For real sessions this is the dense CREStereo depth. It has a lower resolution than the image
(real: 720x1280 vs. 1242x2208, same aspect ratio). The loader therefore resizes the image to the
depth resolution and scales the intrinsics, so image, depth and camera parameters fit together.
"""

import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from blindspot.calibration import scale_intrinsics
from blindspot.core import Intrinsics, Sample

SANPO_URL = "https://storage.googleapis.com/gresearch/sanpo_dataset/v0"
CHEST = "camera_chest"


def read_depth(path: Path) -> np.ndarray:
    """Reads a SANPO depth map.

    Args:
        path: File ``*.float16.gz``; the first two values are the header (height, width).

    Returns:
        Depth in meters as float32 with shape (height, width) from the header; invalid pixels
        (0, inf, NaN) are NaN.

    Raises:
        ValueError: If the pixel count does not match the header.
    """
    with gzip.open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.float16)
    height, width = int(raw[0]), int(raw[1])
    if raw.size != 2 + height * width:
        raise ValueError(f"{path}: {raw.size - 2} pixels, header says {height}x{width}={height * width}")
    depth = raw[2:].reshape(height, width).astype(np.float32)
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return depth


def read_intrinsics(description: dict, camera: str = CHEST) -> tuple[Intrinsics, int, int]:
    """Gets the parameters of the left lens of a camera from ``description.json``.

    Args:
        description: Loaded ``description.json`` of the session.
        camera: Camera name, e.g. ``camera_chest``.

    Returns:
        (Intrinsics, height, width) in pixels.

    Raises:
        ValueError: If the session does not have this camera.
    """
    locations = description["session_camera_location"]
    if camera not in locations:
        raise ValueError(f"Camera {camera} not in session, available: {locations}")
    p = description["session_camera_details"][locations.index(camera)]["left_camera_params"]
    return Intrinsics(p["fx"], p["fy"], p["cx"], p["cy"]), p["image_height"], p["image_width"]


def frame_indices(session_dir: Path, camera: str = CHEST) -> list[int]:
    """Lists the frame numbers for which both image and depth exist (sorted).

    Args:
        session_dir: Directory of a SANPO session.
        camera: Camera name, default is the chest camera.
    """
    left = Path(session_dir) / camera / "left"
    frames = {int(p.stem) for p in (left / "video_frames").glob("*.png")}
    depths = {int(p.name.split(".")[0]) for p in (left / "depth_maps").glob("*.float16.gz")}
    return sorted(frames & depths)


def load_sample(session_dir: Path, index: int, camera: str = CHEST) -> Sample:
    """Loads an image together with depth and camera parameters.

    Args:
        session_dir: Directory of a SANPO session.
        index: Frame number (e.g. 15 for ``000015.png``).
        camera: Camera name, default is the chest camera.

    Returns:
        :class:`Sample` without obstacle annotation (``obstacles=None``). Image and intrinsics are
        converted to the resolution of the depth map.

    Raises:
        FileNotFoundError: If the image is missing.
        ValueError: If the image size does not match ``description.json``.
    """
    session_dir = Path(session_dir)
    description = json.loads((session_dir / "description.json").read_text())
    intrinsics, img_h, img_w = read_intrinsics(description, camera)
    left = session_dir / camera / "left"
    name = f"{index:06d}"
    bgr = cv2.imread(str(left / "video_frames" / f"{name}.png"))
    if bgr is None:
        raise FileNotFoundError(left / "video_frames" / f"{name}.png")
    depth = read_depth(left / "depth_maps" / f"{name}.float16.gz")
    if bgr.shape[:2] != (img_h, img_w):
        raise ValueError(f"Image {bgr.shape[:2]} does not match description.json ({img_h}x{img_w})")
    h, w = depth.shape
    rgb = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return Sample(
        frame_id=f"{session_dir.name}/{camera}/{name}",
        rgb=rgb,
        depth=depth,
        intrinsics=scale_intrinsics(intrinsics, w / img_w, h / img_h),
    )


def read_session_ids(path: Path) -> list[str]:
    """Reads a session list (one ID per line, empty lines are ignored).

    Args:
        path: Text file with session IDs.
    """
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
