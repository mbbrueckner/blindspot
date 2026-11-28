"""Walking corridor and alert logic (WP 3.1): turns a metric depth map into alerts.

Every pixel is back-projected with the intrinsics into camera coordinates (x right, y down, z forward)
and rotated by the camera pitch into a gravity-aligned frame. A pixel is an obstacle point if it lies

    * inside the corridor: lateral offset ``|x| <= half_width_m`` in front of the walker,
    * within ``warn_range_m`` (horizontal distance),
    * between ``min_height_m`` and ``max_height_m`` above the ground, which removes the floor and things
      above head height. The ground height comes from the camera height; there is no ground-plane fit
      yet (WP 2.2), so a wrong pitch makes the floor look like an obstacle or hides low obstacles.

The image is split into thirds (left, center, right, as in the evaluation). A third raises an alert if
it has at least ``min_pixel_fraction`` of its pixels as obstacle points; the alert distance is the
median depth of the obstacle points near the closest one. Multi-frame confirmation and minimum object
size belong to WP 3.4.

Pitch from the iPhone recordings (ARKit poses): median between -4 and +5 degrees per recording, so the
default is 0 (camera looking straight ahead).
"""

import math
from dataclasses import asdict, dataclass

import numpy as np

from blindspot.core import DIRECTIONS, Alert, Intrinsics


@dataclass(frozen=True)
class CorridorConfig:
    """Parameters of the corridor and alert logic.

    Attributes:
        half_width_m: Half width of the corridor (shoulder half width plus margin).
        warn_range_m: Alerts for obstacles up to this horizontal distance.
        camera_height_m: Height of the camera above the ground (chest).
        pitch_deg: Camera tilt, positive = looking down.
        min_height_m: Points lower than this above the ground count as ground.
        max_height_m: Points higher than this (above head height) are ignored.
        min_pixel_fraction: Share of a third's pixels that must be obstacle points to raise an alert.
        near_band_m: Points within this distance behind the closest obstacle point form the obstacle
            whose median depth is reported.
    """

    half_width_m: float = 0.45
    warn_range_m: float = 3.0
    camera_height_m: float = 1.30
    pitch_deg: float = 0.0
    min_height_m: float = 0.15
    max_height_m: float = 2.0
    min_pixel_fraction: float = 0.01
    near_band_m: float = 0.5

    def to_dict(self) -> dict:
        """Plain dict for reports."""
        return asdict(self)


def back_project(depth: np.ndarray, k: Intrinsics) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera coordinates of every pixel.

    Args:
        depth: Depth (z) in meters, shape (H, W).
        k: Intrinsics at the resolution of ``depth``.

    Returns:
        (x, y, z) arrays of shape (H, W): x right, y down, z forward, in meters.
    """
    h, w = depth.shape
    u = np.arange(w, dtype=np.float32)[None, :]
    v = np.arange(h, dtype=np.float32)[:, None]
    x = (u - k.cx) / k.fx * depth
    y = (v - k.cy) / k.fy * depth
    return x, y, depth


def obstacle_mask(depth: np.ndarray, k: Intrinsics, cfg: CorridorConfig) -> np.ndarray:
    """Pixels that are obstacle points inside the walking corridor (see module description).

    Args:
        depth: Metric depth in meters, shape (H, W); NaN or <= 0 is ignored.
        k: Intrinsics at the resolution of ``depth``.
        cfg: Corridor parameters.

    Returns:
        Boolean mask of shape (H, W).
    """
    with np.errstate(invalid="ignore"):
        x, y, z = back_project(depth, k)
    t = math.radians(cfg.pitch_deg)
    with np.errstate(invalid="ignore"):
        forward = z * math.cos(t) - y * math.sin(t)
        height = cfg.camera_height_m - (y * math.cos(t) + z * math.sin(t))
        return (
            np.isfinite(depth)
            & (depth > 0)
            & (np.abs(x) <= cfg.half_width_m)
            & (forward > 0)
            & (forward <= cfg.warn_range_m)
            & (height >= cfg.min_height_m)
            & (height <= cfg.max_height_m)
        )


def alerts_from_mask(
    depth: np.ndarray, mask: np.ndarray, min_pixel_fraction: float, near_band_m: float
) -> list[Alert]:
    """One alert per image third that contains enough masked pixels.

    Args:
        depth: Depth in meters, shape (H, W).
        mask: Obstacle pixels, shape (H, W).
        min_pixel_fraction: Share of the third's pixels needed for an alert.
        near_band_m: Distance band behind the closest point that forms the obstacle.

    Returns:
        Alerts, sorted by distance (nearest first).
    """
    h, w = depth.shape
    alerts = []
    for i, direction in enumerate(DIRECTIONS):
        x0, x1 = round(i * w / 3), round((i + 1) * w / 3)
        d = depth[:, x0:x1][mask[:, x0:x1]]
        if d.size == 0 or d.size < min_pixel_fraction * h * (x1 - x0):
            continue
        nearest = float(np.percentile(d, 2))
        alerts.append(Alert(direction, float(np.median(d[d <= nearest + near_band_m]))))
    return sorted(alerts, key=lambda a: a.distance_m)


def corridor_alerts(depth: np.ndarray, k: Intrinsics, cfg: CorridorConfig = CorridorConfig()) -> list[Alert]:
    """Alerts for obstacles inside the walking corridor.

    Args:
        depth: Metric depth in meters, shape (H, W).
        k: Intrinsics at the resolution of ``depth``.
        cfg: Corridor parameters.

    Returns:
        Alerts sorted by distance, empty if the corridor is free.
    """
    return alerts_from_mask(depth, obstacle_mask(depth, k, cfg), cfg.min_pixel_fraction, cfg.near_band_m)
