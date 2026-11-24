"""Depth Anything 3 (ByteDance Seed). Runs in a separate env (needs Python <= 3.13, numpy < 2, xformers).

Install: see docs/models.md (``da3`` environment).
"""

import numpy as np

from blindspot.core import Intrinsics
from blindspot.models.base import DepthModel, ModelInfo, resize_depth

CANONICAL_FOCAL = 300.0


class DA3(DepthModel):
    """Depth Anything 3 via its ``DepthAnything3.inference`` API.

    The metric model predicts depth for a canonical focal length of 300 px at the network input
    resolution: ``depth_m = focal_input * output / 300`` with ``focal_input`` the mean of fx and fy scaled
    to the processed image. The non-metric models return relative depth (not disparity).

    Args:
        info: Static model description.
        model_id: Hugging Face model ID.
        metric: True for the metric model.
        process_res: Network input resolution (long side), official default 504.
    """

    def __init__(self, info: ModelInfo, model_id: str, metric: bool, process_res: int = 504) -> None:
        """Stores the configuration; weights are loaded in :meth:`load`."""
        super().__init__(info)
        self.model_id = model_id
        self.metric = metric
        self.process_res = process_res

    def load(self, device: str) -> None:
        """Downloads (once) and loads the model."""
        from depth_anything_3.api import DepthAnything3

        self.device = device
        self.module = DepthAnything3.from_pretrained(self.model_id).to(device).eval()

    def predict(self, rgb: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
        """Predicts depth at the input resolution (meters if metric, otherwise relative depth)."""
        h, w = rgb.shape[:2]
        pred = self.module.inference([rgb], process_res=self.process_res)
        depth = np.asarray(pred.depth[0], dtype=np.float32)
        if self.metric:
            ph, pw = depth.shape
            focal = 0.5 * (intrinsics.fx * pw / w + intrinsics.fy * ph / h)
            depth = depth * focal / CANONICAL_FOCAL
        return resize_depth(depth, (h, w))
