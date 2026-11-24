"""Derives obstacle annotations from SANPO panoptic masks (WP 1.4).

A mask PNG has three channels: the first is the class ID (see SANPO ``labelmap.json``), the second
and third form the instance ID (``second * 256 + third``). Each instance of an obstacle class
becomes one :class:`~blindspot.core.Obstacle` with its bounding rectangle. Which classes count as
obstacles are our own decision, set by the constants below. "In the walking path" is decided in 3D: an
obstacle is in the path if enough of its pixels lie inside the corridor of WP 3.1, computed from the
reference depth. The range is left open here; the evaluation applies the warning range.
"""

import math
from dataclasses import replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from blindspot.alerts.corridor import CorridorConfig, obstacle_mask
from blindspot.core import Intrinsics, Obstacle
from blindspot.data.sanpo import CHEST

# SANPO class ID -> label for classes with instance IDs (one rectangle per instance). Ground, sky and
# vegetation are not boxed obstacles. Revisit with the group if the definition should change.
OBSTACLE_LABELS = {
    12: "pedestrian",
    13: "rider",
    14: "animal",
    20: "obstacle",
    21: "vehicle",
    22: "traffic sign",
    23: "traffic light",
    24: "pole",
    25: "bus stop",
    26: "bike rack",
    28: "tree",
}

# Classes without instance IDs: every connected region becomes one rectangle.
SEMANTIC_OBSTACLE_LABELS = {4: "barrier", 7: "building", 8: "wall/fence", 9: "hand rail"}

# Connected regions need this many times ``min_area_px``, smaller patches are mask noise.
SEMANTIC_AREA_FACTOR = 10


def obstacles_from_mask(
    mask: np.ndarray, corridor: np.ndarray, min_area_px: int = 400, min_corridor_px: int = 100
) -> list[Obstacle]:
    """Converts a panoptic mask into obstacle rectangles.

    Args:
        mask: Mask image, shape (H, W, 3), uint8, channels in the order class, instance high, instance low.
        corridor: Boolean (H, W) pixels inside the walking corridor; its shape is the target size, the
            mask is resized with nearest neighbor if it differs.
        min_area_px: Instances with fewer mask pixels (at the target size) are dropped as too small.
        min_corridor_px: An obstacle is ``in_path`` if at least this many of its own mask pixels (not
            the box) are corridor pixels. This ignores single noisy depth pixels.

    Returns:
        One obstacle per instance of an obstacle class and per large connected region of a semantic
        class (building, wall).
    """
    h, w = corridor.shape
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    semantic = mask[:, :, 0]
    instance = mask[:, :, 1].astype(np.int32) * 256 + mask[:, :, 2]
    obstacles = []
    for class_id, label in OBSTACLE_LABELS.items():
        of_class = semantic == class_id
        for inst in np.unique(instance[of_class]):
            region = of_class & (instance == inst)
            ys, xs = np.nonzero(region)
            if ys.size < min_area_px:
                continue
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            obstacles.append(
                Obstacle(box, label, bool(np.count_nonzero(corridor[region]) >= min_corridor_px))
            )
    for class_id, label in SEMANTIC_OBSTACLE_LABELS.items():
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (semantic == class_id).astype(np.uint8), connectivity=8
        )
        for i in range(1, n):  # label 0 is the background
            x, y, bw, bh, area = stats[i]
            if area >= SEMANTIC_AREA_FACTOR * min_area_px:
                box = (int(x), int(y), int(x + bw), int(y + bh))
                in_corridor = np.count_nonzero(corridor[labels == i]) >= min_corridor_px
                obstacles.append(Obstacle(box, label, bool(in_corridor)))
    return obstacles


def load_obstacles(
    session_dir: Path,
    index: int,
    depth: np.ndarray,
    k: Intrinsics,
    camera: str = CHEST,
    min_area_px: int = 400,
    min_corridor_px: int = 100,
    cfg: CorridorConfig = CorridorConfig(),
) -> Optional[list[Obstacle]]:
    """Reads the mask of one frame and converts it to obstacles.

    Args:
        session_dir: Directory of a SANPO session.
        index: Frame number (e.g. 15 for ``000015.png``).
        depth: Reference depth of the frame in meters, shape (H, W); defines the target size.
        k: Intrinsics at the resolution of ``depth``.
        camera: Camera name, default is the chest camera.
        min_area_px: Smallest instance in mask pixels, see :func:`obstacles_from_mask`.
        min_corridor_px: Corridor pixels needed for ``in_path``, see :func:`obstacles_from_mask`.
        cfg: Corridor parameters; ``warn_range_m`` is ignored (no range limit).

    Returns:
        Obstacle list, or ``None`` if the frame has no mask (not annotated).
    """
    path = Path(session_dir) / camera / "left" / "segmentation_masks" / f"{index:06d}.png"
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    corridor = obstacle_mask(depth, k, replace(cfg, warn_range_m=math.inf))
    return obstacles_from_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), corridor, min_area_px, min_corridor_px)
