"""MoGe-2 (Microsoft, metric depth from a point map; uses the field of view). Runs in a separate env.

Install: see docs/models.md (``moge`` environment).
"""

import math

import numpy as np

from blindspot.core import Intrinsics
from blindspot.models.base import DepthModel, ModelInfo, resize_depth


class MoGe2(DepthModel):
    """MoGe-2 model; we pass our horizontal field of view so it does not have to estimate it.

    Args:
        info: Static model description.
        model_id: Hugging Face model ID.
    """

    def __init__(self, info: ModelInfo, model_id: str) -> None:
        """Stores the configuration; weights are loaded in :meth:`load`."""
        super().__init__(info)
        self.model_id = model_id

    def load(self, device: str) -> None:
        """Downloads (once) and loads the model."""
        from moge.model.v2 import MoGeModel

        self.device = device
        self.module = MoGeModel.from_pretrained(self.model_id).to(device).eval()

    def predict(self, rgb: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
        """Predicts metric depth in meters at the input resolution (NaN where MoGe masks the output)."""
        import torch

        h, w = rgb.shape[:2]
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float() / 255.0
        fov_x = math.degrees(2 * math.atan(w / (2 * intrinsics.fx)))
        with torch.inference_mode():
            out = self.module.infer(x, fov_x=fov_x, use_fp16=self.device == "cuda")
        depth = out["depth"].float().cpu().numpy()
        depth[~np.isfinite(depth)] = np.nan
        return resize_depth(depth, (h, w))
