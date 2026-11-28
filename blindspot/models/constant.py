"""Constant-depth baseline: a sanity floor that every real model has to beat (WP 5.1)."""

import numpy as np

from blindspot.core import Intrinsics
from blindspot.models.base import DepthModel, ModelInfo


class ConstantDepth(DepthModel):
    """Predicts the same metric depth everywhere (no weights, no torch).

    Args:
        info: Static model description.
        depth_m: Predicted depth in meters.
    """

    def __init__(self, info: ModelInfo, depth_m: float = 6.0) -> None:
        """Stores the description and the constant."""
        super().__init__(info)
        self.depth_m = depth_m

    def load(self, device: str) -> None:
        """Nothing to load."""
        self.device = device

    def predict(self, rgb: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
        """Returns ``depth_m`` for every pixel (float32, shape (H, W))."""
        return np.full(rgb.shape[:2], self.depth_m, np.float32)
