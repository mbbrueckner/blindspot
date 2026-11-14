"""Calibrates the iPhone camera from chessboard photos and stores the result in the repo (WP 1.1).

Example (board with 9x6 inner corners, 25 mm squares, camera 1.35 m above the ground)::

    python scripts/calibrate.py photos/ --pattern 9 6 --square-mm 25 --height-m 1.35

The photo format must match later operation (same resolution, same lens, no zoom).
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blindspot.calibration import calibrate_from_corners, find_corners, save_calibration  # noqa: E402


def main() -> None:
    """Reads the photos, calibrates and writes ``planning/calibration/camera.json``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--pattern", type=int, nargs=2, required=True, metavar=("COLS", "ROWS"))
    ap.add_argument("--square-mm", type=float, required=True)
    ap.add_argument("--height-m", type=float, required=True, help="measured camera height above the ground")
    args = ap.parse_args()

    paths = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"})
    images = [im for im in (cv2.imread(str(p)) for p in paths) if im is not None]
    if not images:
        sys.exit(f"No readable images in {args.folder} (convert HEIC to JPG first)")
    size = {(im.shape[1], im.shape[0]) for im in images}
    if len(size) != 1:
        sys.exit(f"Images have different sizes: {size}")
    corners = find_corners(images, tuple(args.pattern))
    print(f"Board found in {len(corners)} of {len(images)} images")
    k, rms = calibrate_from_corners(corners, tuple(args.pattern), args.square_mm / 1000, size.pop())
    print(f"{k}, RMS {rms:.2f} px")
    save_calibration(
        k,
        (images[0].shape[1], images[0].shape[0]),
        args.height_m,
        f"chessboard, {len(corners)} images",
    )


if __name__ == "__main__":
    main()
