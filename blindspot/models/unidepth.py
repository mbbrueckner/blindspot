"""UniDepthV2 (metric depth, uses the camera intrinsics). Runs in a separate env.

Install: see docs/models.md (``unidepth`` environment; UniDepth pins torch and needs xformers).
"""

import numpy as np

from blindspot.core import Intrinsics
from blindspot.models.base import DepthModel, ModelInfo, resize_depth


class UniDepth2(DepthModel):
    """UniDepthV2; the pinhole intrinsics at the input resolution are passed in, the model rescales them.

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
        from unidepth.models import UniDepthV2

        self.device = device
        self.module = UniDepthV2.from_pretrained(self.model_id).to(device).eval()

    def predict(self, rgb: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
        """Predicts metric depth in meters at the input resolution."""
        import torch
        from unidepth.utils.camera import Pinhole

        k = torch.tensor(
            [[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]],
            dtype=torch.float32,
        )
        x = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device)
        with torch.inference_mode():
            pred = self.module.infer(x, Pinhole(K=k[None].to(self.device)))
        return resize_depth(pred["depth"].squeeze().float().cpu().numpy(), rgb.shape[:2])
