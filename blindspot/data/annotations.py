"""Obstacle annotation (WP 1.4): reads the per-frame obstacle boxes from JSON."""

import json

from blindspot.core import Obstacle


def load_annotations(path: str) -> dict[str, list[Obstacle]]:
    """Reads the obstacle annotation from a JSON file (WP 1.4).

    Format: ``{"<frame_id>": [{"box": [x0, y0, x1, y1], "label": "pole", "in_path": true}, ...]}``.
    An empty list means "annotated, no obstacle"; missing frames count as not annotated.

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
