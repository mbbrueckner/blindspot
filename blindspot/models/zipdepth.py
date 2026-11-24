"""ZipDepth (ECCV 2026, lightweight, relative disparity), from the official GitHub repository."""

import numpy as np

from blindspot.core import Intrinsics
from blindspot.models.base import DepthModel, ModelInfo, disparity_to_pseudo_depth
from blindspot.models.registry import weights_dir

COMMIT = "91f3fd21e131641f51e8d35736d1958350180e3a"
CHECKPOINT_URL = f"https://github.com/fabiotosi92/ZipDepth/raw/{COMMIT}/checkpoints/zipdepth_base.pth"


class ZipDepth(DepthModel):
    """ZipDepth base model with the preprocessing of ``zipdepth.inference.predictor``.

    The image is resized so its shorter side is ``input_size`` (both sides multiples of 32) and scaled to
    [0, 1]; the network normalizes internally. Reparameterizable blocks are fused for inference.

    Args:
        info: Static model description.
        input_size: Shorter side of the network input in pixels (official default 384).
    """

    def __init__(self, info: ModelInfo, input_size: int = 384) -> None:
        """Stores the configuration; weights are loaded in :meth:`load`."""
        super().__init__(info)
        self.input_size = input_size

    def load(self, device: str) -> None:
        """Downloads the checkpoint once (pinned commit), builds, loads and fuses the model.

        Raises:
            RuntimeError: If the checkpoint does not fit the architecture (missing weights).
        """
        import torch
        from zipdepth.model.architecture import create_model
        from zipdepth.utils.model_utils import strip_state_dict_prefixes

        self.device = device
        path = weights_dir() / "zipdepth_base.pth"
        if not path.is_file():
            torch.hub.download_url_to_file(CHECKPOINT_URL, str(path))
        model = create_model(variant="base", upsample_unfold=True)
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = strip_state_dict_prefixes(state.get("model_state_dict", state))
        missing, _ = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"ZipDepth checkpoint is missing {len(missing)} weights, e.g. {missing[:3]}")
        self.module = model.to(device).eval()
        self.module.fuse_for_inference()

    def predict(self, rgb: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
        """Predicts relative disparity and returns pseudo-depth at the input resolution."""
        import cv2
        import torch
        import torch.nn.functional as F

        h, w = rgb.shape[:2]
        scale = self.input_size / min(h, w)
        new_h, new_w = (max(32, int(round(v * scale / 32)) * 32) for v in (h, w))
        x = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(x).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        with torch.inference_mode():
            disp = self.module(x)
            disp = disp.reshape(1, 1, *disp.shape[-2:])
            disp = F.interpolate(disp, (h, w), mode="bilinear", align_corners=True)
        return disparity_to_pseudo_depth(disp[0, 0].float().cpu().numpy())
