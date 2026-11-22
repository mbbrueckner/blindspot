"""Derives obstacle annotations from SANPO panoptic masks (WP 1.4).

A mask PNG has three channels: the first is the class ID (see SANPO ``labelmap.json``), the second
and third form the instance ID (``second * 256 + third``). Each instance of an obstacle class
becomes one :class:`~blindspot.core.Obstacle` with its bounding rectangle. Which classes count as
obstacles and what "in the walking path" means are our own decisions, set by the constants below.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from blindspot.core import Obstacle
from blindspot.sanpo import CHEST

# SANPO class ID -> label. Ground, buildings, walls, sky and vegetation are not boxed obstacles;
# traffic lights hang above head height. Revisit with the group if the definition should change.
OBSTACLE_LABELS = {
    12: "pedestrian",
    13: "rider",
    14: "animal",
    20: "obstacle",
    21: "vehicle",
    22: "traffic sign",
    24: "pole",
    25: "bus stop",
    26: "bike rack",
    28: "tree",
}


def in_path(box: tuple[int, int, int, int], width: int) -> bool:
    """Tells whether a rectangle overlaps the central third of the image width.

    This is a deliberately simple 2D stand-in for the walking corridor.

    Args:
        box: (x0, y0, x1, y1) in pixels.
        width: Image width in pixels.

    Returns:
        True if the rectangle reaches into the central third.
    """
    return box[0] < 2 * width / 3 and box[2] > width / 3


def obstacles_from_mask(mask: np.ndarray, size: tuple[int, int], min_area_px: int = 400) -> list[Obstacle]:
    """Converts a panoptic mask into obstacle rectangles.

    Args:
        mask: Mask image, shape (H, W, 3), uint8, channels in the order class, instance high, instance low.
        size: Target (height, width) in pixels; the mask is resized with nearest neighbor if it differs.
        min_area_px: Instances with fewer mask pixels (at the target size) are dropped as too small.

    Returns:
        One obstacle per instance of an obstacle class, with ``in_path`` from :func:`in_path`.
    """
    h, w = size
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    semantic = mask[:, :, 0]
    instance = mask[:, :, 1].astype(np.int32) * 256 + mask[:, :, 2]
    obstacles = []
    for class_id, label in OBSTACLE_LABELS.items():
        of_class = semantic == class_id
        for inst in np.unique(instance[of_class]):
            ys, xs = np.nonzero(of_class & (instance == inst))
            if ys.size < min_area_px:
                continue
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            obstacles.append(Obstacle(box, label, in_path(box, w)))
    return obstacles


def load_obstacles(
    session_dir: Path, index: int, size: tuple[int, int], camera: str = CHEST, min_area_px: int = 400
) -> Optional[list[Obstacle]]:
    """Reads the mask of one frame and converts it to obstacles.

    Args:
        session_dir: Directory of a SANPO session.
        index: Frame number (e.g. 15 for ``000015.png``).
        size: Target (height, width) in pixels, normally the shape of the loaded sample.
        camera: Camera name, default is the chest camera.
        min_area_px: Smallest instance in mask pixels, see :func:`obstacles_from_mask`.

    Returns:
        Obstacle list, or ``None`` if the frame has no mask (not annotated).
    """
    path = Path(session_dir) / camera / "left" / "segmentation_masks" / f"{index:06d}.png"
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    return obstacles_from_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size, min_area_px)
