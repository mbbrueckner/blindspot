"""Obstacle annotation (WP 1.4): reads the per-frame obstacle boxes from JSON and attaches them."""

import json
from typing import Iterable

from blindspot.core import Obstacle, Sample


def load_annotations(path: str) -> dict[str, list[Obstacle]]:
    """Reads the obstacle annotation from a JSON file (WP 1.4).

    Format: ``{"<frame_id>": [{"box": [x0, y0, x1, y1], "label": "pole", "in_path": true}, ...]}``.
    An empty list means "annotated, no obstacle"; missing frames count as not annotated.
    Box coordinates are pixels of the loaded :class:`Sample` image (for SANPO: the depth resolution).

    Args:
        path: Path to the JSON file.

    Returns:
        Dict mapping frame_id to a list of obstacles.
    """
    with open(path) as f:
        raw = json.load(f)
    return {
        fid: [Obstacle(tuple(o["box"]), o["label"], bool(o["in_path"])) for o in obs]
        for fid, obs in raw.items()
    }


def apply_annotations(samples: Iterable[Sample], annotations: dict[str, list[Obstacle]]) -> int:
    """Sets ``Sample.obstacles`` from the annotation, matched by ``frame_id``.

    Samples without an entry keep their ``obstacles`` (``None`` = not annotated).

    Args:
        samples: Samples to annotate, modified in place.
        annotations: Dict as returned by :func:`load_annotations`.

    Returns:
        Number of samples that received an annotation.

    Raises:
        ValueError: If a box is empty or lies outside the image, because such a box would silently
            yield no distance and the obstacle would never count.
    """
    n = 0
    for s in samples:
        if s.frame_id not in annotations:
            continue
        h, w = s.rgb.shape[:2]
        for ob in annotations[s.frame_id]:
            x0, y0, x1, y1 = ob.box
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError(f"Box {ob.box} of {s.frame_id} is outside the {w}x{h} image")
        s.obstacles = annotations[s.frame_id]
        n += 1
    return n
