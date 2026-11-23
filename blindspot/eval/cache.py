"""On-disk cache of model predictions, so metrics, reports and plots can be recomputed without the models.

Layout::

    <root>/<model>/<frame_id>.npy   raw model output (float16, model resolution, meters or pseudo-depth)
    <root>/<model>/meta.json        model info, cache version, input size
    <root>/<model>/perf_<device>.json  load time, parameters, latency and memory of the last run

``frame_id`` contains slashes (``<recording>/000123``), which become sub-folders. float16 keeps 3
significant digits, about 0.1 % relative error, which is far below the model errors we measure.
The cache does not know when model code changes: bump ``cache_version`` in the model info or run with
``--recompute``.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_ROOT = Path("results/cache")


class PredictionCache:
    """Reads and writes cached predictions of one cache root.

    Args:
        root: Cache directory, default ``results/cache``.
    """

    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        """Stores the cache root."""
        self.root = Path(root)

    def path(self, model: str, frame_id: str) -> Path:
        """File of one prediction.

        Raises:
            ValueError: If ``frame_id`` would escape the model folder.
        """
        if ".." in Path(frame_id).parts or Path(frame_id).is_absolute():
            raise ValueError(f"Invalid frame id {frame_id!r}")
        return self.root / model / f"{frame_id}.npy"

    def has(self, model: str, frame_id: str) -> bool:
        """True if a prediction for this model and frame is cached."""
        return self.path(model, frame_id).is_file()

    def load(self, model: str, frame_id: str) -> np.ndarray:
        """Loads a cached prediction as float32.

        Raises:
            FileNotFoundError: If it is not cached.
        """
        return np.load(self.path(model, frame_id)).astype(np.float32)

    def save(self, model: str, frame_id: str, pred: np.ndarray) -> None:
        """Stores a prediction as float16 (values beyond the float16 range are clipped)."""
        path = self.path(model, frame_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        finfo = np.finfo(np.float16)
        np.save(path, np.clip(pred, finfo.min, finfo.max).astype(np.float16))

    def read_json(self, model: str, name: str) -> Optional[dict]:
        """Reads ``<root>/<model>/<name>.json``, ``None`` if missing."""
        path = self.root / model / f"{name}.json"
        return json.loads(path.read_text()) if path.is_file() else None

    def write_json(self, model: str, name: str, data: dict) -> None:
        """Writes ``<root>/<model>/<name>.json``."""
        path = self.root / model / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str) + "\n")

    def invalidate_if_stale(self, model: str, meta: dict) -> bool:
        """Deletes the model's cached predictions if the stored meta differs from ``meta``.

        Only the keys in ``meta`` are compared; additional keys in the stored meta (e.g. descriptive fields
        written by an older version) are ignored.

        Args:
            model: Registry name.
            meta: Current model meta (the fields that determine the predictions).

        Returns:
            True if the cache was cleared.
        """
        old = self.read_json(model, "meta")
        if old is None or _matches(old, json.loads(json.dumps(meta, default=str))):
            return False
        self.clear(model)
        return True

    def clear(self, model: str) -> None:
        """Deletes all cached predictions, meta and perf files of a model."""
        folder = self.root / model
        if not folder.exists():
            return
        for p in sorted(folder.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        folder.rmdir()


def _matches(old, new) -> bool:
    """True if every key of ``new`` exists in ``old`` with an equal value (recursively for dicts)."""
    if isinstance(new, dict):
        return isinstance(old, dict) and all(k in old and _matches(old[k], v) for k, v in new.items())
    return old == new
