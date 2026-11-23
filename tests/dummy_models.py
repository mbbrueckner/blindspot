"""Dummy depth models for the tests (no torch, no weights).

They read the depth from the image: the test samples encode the reference depth in the red channel
(``red = 20 * depth``), so a model can be "perfect" or off by a known factor without seeing the ground truth.
"""

import numpy as np

from blindspot.models.base import DepthModel, disparity_to_pseudo_depth


def decode(rgb: np.ndarray) -> np.ndarray:
    return rgb[:, :, 0].astype(np.float32) / 20.0


class ScaledDepth(DepthModel):
    """Metric dummy: depth times ``factor``."""

    def __init__(self, info, factor=1.0):
        super().__init__(info)
        self.factor = factor

    def load(self, device):
        self.device = device

    def predict(self, rgb, intrinsics):
        return decode(rgb) * self.factor


class RelativeDisparity(DepthModel):
    """Relative dummy: affine disparity ``3 / depth + 0.5``, returned as pseudo-depth."""

    def __init__(self, info):
        super().__init__(info)

    def load(self, device):
        self.device = device

    def predict(self, rgb, intrinsics):
        return disparity_to_pseudo_depth(3.0 / decode(rgb) + 0.5)


class BrokenLoad(DepthModel):
    """Fails when loading weights."""

    def __init__(self, info):
        super().__init__(info)

    def load(self, device):
        raise RuntimeError("weights not found")

    def predict(self, rgb, intrinsics):  # pragma: no cover - never reached
        raise AssertionError


class WrongShape(DepthModel):
    """Returns a prediction of the wrong size."""

    def __init__(self, info):
        super().__init__(info)

    def load(self, device):
        self.device = device

    def predict(self, rgb, intrinsics):
        return np.ones((2, 2), np.float32)
