"""Common interface of all depth models.

A model is loaded once (:meth:`DepthModel.load`) and then predicts one image at a time
(:meth:`DepthModel.predict`). Every model returns depth at the resolution of the input image:

    * metric models: depth in meters;
    * relative models: *pseudo-depth* = 1 / disparity, with the disparity divided by its median first,
      so the values are around 1. Only the ordering and ratios mean something; scoring needs alignment
      (see :mod:`blindspot.eval.alignment`).

Models get the camera intrinsics of the input image. Models that use the focal length (Metric3D,
Depth Pro, MoGe, UniDepth, DA3 Metric) scale them to their own input resolution. Models never see
the ground truth.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np

from blindspot.core import Intrinsics


@dataclass(frozen=True)
class ModelInfo:
    """Static description of a model (available without importing torch).

    Attributes:
        name: Short registry name, e.g. "da2_s".
        label: Display name, e.g. "Depth Anything V2 Small".
        family: Model family, e.g. "Depth Anything V2".
        kind: "metric" (meters) or "relative" (pseudo-depth, needs alignment).
        uses_intrinsics: True if the model uses the focal length we pass in.
        source: Weights (Hugging Face ID, torch hub repo or URL), pinned where possible.
        install: How to get the dependencies, e.g. "extra: models" or a pip command.
        env: "main" if it runs in the project environment, otherwise the name of a separate environment.
        cache_version: Bump when the wrapper changes its output, so cached predictions are recomputed.
        notes: Anything special, e.g. input resolution.
    """

    name: str
    label: str
    family: str
    kind: str
    uses_intrinsics: bool
    source: str
    install: str = "extra: models"
    env: str = "main"
    cache_version: int = 1
    notes: str = ""

    def to_dict(self) -> dict:
        """Plain dict for JSON."""
        return asdict(self)


class DepthModel(ABC):
    """Base class of the model wrappers.

    Subclasses set ``self.module`` (the torch module, used for counting parameters) in :meth:`load`.

    Args:
        info: Static model description.
    """

    def __init__(self, info: ModelInfo) -> None:
        """Stores the static description."""
        self.info = info
        self.device = "cpu"
        self.module = None

    @abstractmethod
    def load(self, device: str) -> None:
        """Loads weights and moves the model to ``device`` ("cuda", "mps" or "cpu")."""

    @abstractmethod
    def predict(self, rgb: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
        """Predicts depth for one image.

        Args:
            rgb: Image, RGB uint8, shape (H, W, 3).
            intrinsics: Camera parameters at the resolution of ``rgb``.

        Returns:
            float32 array of shape (H, W): meters (metric) or pseudo-depth (relative), NaN where invalid.
        """

    def n_params(self) -> Optional[int]:
        """Number of parameters, ``None`` if unknown."""
        from blindspot.eval.timing import count_parameters

        return count_parameters(self.module)

    def unload(self) -> None:
        """Drops the model so its memory can be freed."""
        self.module = None


def disparity_to_pseudo_depth(disparity: np.ndarray) -> np.ndarray:
    """Converts relative disparity to pseudo-depth (see module description).

    Args:
        disparity: Relative disparity (larger = nearer), any scale.

    Returns:
        float32 pseudo-depth ``median(d) / d``; NaN where the disparity is not positive.
    """
    d = disparity.astype(np.float64)
    positive = np.isfinite(d) & (d > 0)
    if not positive.any():
        return np.full(d.shape, np.nan, np.float32)
    d = d / np.median(d[positive])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(positive & (d > 1e-6), 1.0 / d, np.nan).astype(np.float32)


def resize_depth(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resizes a depth or disparity map to (H, W) with bilinear interpolation (float32)."""
    if depth.shape == tuple(shape):
        return depth.astype(np.float32, copy=False)
    return cv2.resize(depth.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def scaled_intrinsics(k: Intrinsics, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> Intrinsics:
    """Intrinsics of an image resized from ``src_hw`` to ``dst_hw`` (both (H, W)).

    Args:
        k: Intrinsics at ``src_hw``.
        src_hw: Original (H, W).
        dst_hw: Resized (H, W).
    """
    sx, sy = dst_hw[1] / src_hw[1], dst_hw[0] / src_hw[0]
    return Intrinsics(k.fx * sx, k.fy * sy, k.cx * sx, k.cy * sy)
