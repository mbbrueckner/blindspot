"""MiDaS via torch hub (relative disparity)."""

import numpy as np

from blindspot.core import Intrinsics
from blindspot.models.base import DepthModel, ModelInfo, disparity_to_pseudo_depth

HUB_REPO = "intel-isl/MiDaS:v3_1"
BACKBONE_REPO = "rwightman/gen-efficientnet-pytorch"  # loaded by MiDaS_small itself (default branch)


class MiDaS(DepthModel):
    """MiDaS model from the official torch hub entry, with its official input transform.

    Args:
        info: Static model description.
        hub_model: Hub entry point, e.g. "MiDaS_small" or "DPT_Hybrid".
    """

    def __init__(self, info: ModelInfo, hub_model: str) -> None:
        """Stores the configuration; weights are loaded in :meth:`load`."""
        super().__init__(info)
        self.hub_model = hub_model
        self.transform = None

    def load(self, device: str) -> None:
        """Loads model and transform from torch hub (downloads once into the torch hub cache)."""
        import torch

        self.device = device
        # MiDaS_small builds its EfficientNet backbone with a nested torch.hub.load that does not pass
        # trust_repo, so torch would wait for a yes/no on stdin. Trusting the repo up front avoids that.
        torch.hub.list(BACKBONE_REPO, trust_repo=True, verbose=False)
        self.module = torch.hub.load(HUB_REPO, self.hub_model, trust_repo=True).to(device).eval()
        transforms = torch.hub.load(HUB_REPO, "transforms", trust_repo=True)
        self.transform = (
            transforms.small_transform if "small" in self.hub_model.lower() else transforms.dpt_transform
        )

    def predict(self, rgb: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
        """Predicts relative disparity and returns pseudo-depth at the input resolution."""
        import torch

        h, w = rgb.shape[:2]
        x = self.transform(rgb).to(self.device)
        with torch.inference_mode():
            disp = self.module(x)
            disp = torch.nn.functional.interpolate(
                disp[:, None], size=(h, w), mode="bicubic", align_corners=False
            )
        return disparity_to_pseudo_depth(disp[0, 0].float().cpu().numpy())
