"""Shared data types for samples, annotations and alerts.

All modules (SANPO loader, own recordings, evaluation) exchange data in these types.
Convention: depth in meters as float32, invalid pixels are NaN; images are RGB uint8 (H, W, 3).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

DIRECTIONS = ("left", "center", "right")


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole camera parameters, all values in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class Obstacle:
    """An annotated obstacle in an image (WP 1.4).

    Attributes:
        box: Rectangle (x0, y0, x1, y1) in pixels.
        label: Obstacle type, e.g. "pole" or "branch".
        in_path: True if the obstacle lies in the walking corridor and should trigger an alert.
    """

    box: tuple[int, int, int, int]
    label: str
    in_path: bool


@dataclass(frozen=True)
class Alert:
    """A warning that a variant emits for one image (WP 5.1).

    Attributes:
        direction: "left", "center" or "right".
        distance_m: Estimated distance to the obstacle in meters.
    """

    direction: str
    distance_m: float


@dataclass
class Sample:
    """A single image with reference depth.

    Attributes:
        frame_id: Unique identifier, e.g. "<session>/camera_chest/000015".
        rgb: Image, RGB uint8, shape (H, W, 3).
        depth: Reference depth in meters, float32, shape (H, W), NaN where invalid.
        intrinsics: Camera parameters matching the resolution of ``rgb``.
        obstacles: Annotation; ``None`` means "not annotated", an empty list means "no obstacle".
    """

    frame_id: str
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: Intrinsics
    obstacles: Optional[list[Obstacle]] = field(default=None)
