"""Camera calibration (WP 1.1): intrinsics from chessboard photos, stored in the repo.

The camera height above the ground cannot be computed; it is measured on the body and entered by
hand. It is the basis of the ground scaling (WP 2.2).
"""

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from blindspot.core import Intrinsics

CALIBRATION_FILE = Path("planning/calibration/camera.json")


def scale_intrinsics(k: Intrinsics, sx: float, sy: float) -> Intrinsics:
    """Converts intrinsics to another resolution (e.g. 1920x1440 video to 256x192 LiDAR depth).

    Args:
        k: Parameters in pixels of the original resolution.
        sx: Factor target width / original width.
        sy: Factor target height / original height.

    Returns:
        Intrinsics in pixels of the target resolution.
    """
    return Intrinsics(k.fx * sx, k.fy * sy, k.cx * sx, k.cy * sy)


def find_corners(images: list[np.ndarray], pattern: tuple[int, int]) -> list[np.ndarray]:
    """Finds the inner chessboard corners in grayscale or BGR images.

    Args:
        images: Photos of the chessboard.
        pattern: Number of inner corners (columns, rows), e.g. (9, 6).

    Returns:
        Corner points (N, 2) float32 for each image in which the board was found; other images
        are left out.
    """
    found = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        ok, corners = cv2.findChessboardCorners(gray, pattern)
        if ok:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            found.append(corners.reshape(-1, 2))
    return found


def calibrate_from_corners(
    corners: list[np.ndarray],
    pattern: tuple[int, int],
    square_m: float,
    image_size: tuple[int, int],
) -> tuple[Intrinsics, float]:
    """Determines focal length and principal point from detected chessboard corners.

    Args:
        corners: Corner points (N, 2) per photo, see :func:`find_corners`; at least 5 photos.
        pattern: Inner corners (columns, rows).
        square_m: Edge length of one square in meters.
        image_size: (width, height) of the photos in pixels.

    Returns:
        (Intrinsics, RMS reprojection error in pixels). Below 1 pixel is good.

    Raises:
        ValueError: If there are fewer than 5 photos.
    """
    if len(corners) < 5:
        raise ValueError(f"Only {len(corners)} photos, at least 5 needed")
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * square_m
    rms, mtx, _, _, _ = cv2.calibrateCamera(
        [obj] * len(corners),
        [c.astype(np.float32).reshape(-1, 1, 2) for c in corners],
        image_size,
        None,
        None,
    )
    return Intrinsics(mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]), float(rms)


def save_calibration(
    k: Intrinsics,
    image_size: tuple[int, int],
    camera_height_m: float,
    source: str,
    path: Path = CALIBRATION_FILE,
) -> None:
    """Stores the calibration as JSON in the repo.

    Args:
        k: Intrinsics in pixels of the resolution ``image_size``.
        image_size: (width, height) in pixels.
        camera_height_m: Measured height of the camera above the ground in meters.
        source: Where the values come from, e.g. "chessboard, 12 images" or "ARKit".
        path: Target file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        **{k_: float(v) for k_, v in asdict(k).items()},
        "image_width": image_size[0],
        "image_height": image_size[1],
        "camera_height_m": camera_height_m,
        "source": source,
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_calibration(path: Path = CALIBRATION_FILE) -> tuple[Intrinsics, tuple[int, int], float]:
    """Reads the calibration.

    Args:
        path: Calibration file written by :func:`save_calibration`.

    Returns:
        (Intrinsics, (width, height), camera height in meters).
    """
    d = json.loads(Path(path).read_text())
    return (
        Intrinsics(d["fx"], d["fy"], d["cx"], d["cy"]),
        (d["image_width"], d["image_height"]),
        d["camera_height_m"],
    )
