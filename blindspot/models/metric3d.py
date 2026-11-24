"""Metric3D v2 via torch hub (metric depth, uses the focal length).

The hub code assumes CUDA and mmcv. Two compatibility fixes make it run on CUDA, MPS and CPU without
mmcv (both applied in :func:`_patch_hub_code`):

* ``mono/utils/comm.py`` imports ``mmcv.utils.collect_env`` unconditionally (only used for logging).
  If mmcv is missing, a stub module is registered before the import; the config loader then falls back
  to ``mmengine`` on its own (that fallback is part of the hub code).
* ``RAFTDepthNormalDPT5.get_bins`` creates its depth bins with ``device="cuda"``. It is replaced by a
  version that uses the device of the model weights.

xFormers is optional in the hub code (it falls back to plain attention).
"""

import math
import sys
import types

import cv2
import numpy as np

from blindspot.core import Intrinsics
from blindspot.models.base import DepthModel, ModelInfo, resize_depth, scaled_intrinsics

HUB_REPO = "yvanyin/metric3d:eb5b6fac0dc155e4e52f576e304fbf11655ff339"
INPUT_SIZE = (616, 1064)  # (H, W) for the ViT models
CANONICAL_FOCAL = 1000.0
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


def _stub_mmcv() -> None:
    """Registers a minimal ``mmcv.utils`` with ``collect_env`` if mmcv is not installed."""
    try:
        import mmcv.utils  # noqa: F401
    except ImportError:
        mmcv = types.ModuleType("mmcv")
        utils = types.ModuleType("mmcv.utils")
        utils.collect_env = lambda: {}
        mmcv.utils = utils
        sys.modules.setdefault("mmcv", mmcv)
        sys.modules.setdefault("mmcv.utils", utils)


def _patch_hub_code() -> None:
    """Makes the depth bins follow the model device (see module description)."""
    import torch

    decoder = sys.modules.get("mono.model.decode_heads.RAFTDepthNormalDPTDecoder5")
    if decoder is None or getattr(decoder.RAFTDepthNormalDPT5, "_blindspot_patched", False):
        return

    def get_bins(self, bins_num):
        device = next(self.parameters()).device
        bins = torch.linspace(math.log(self.min_val), math.log(self.max_val), bins_num, device=device)
        return torch.exp(bins)

    decoder.RAFTDepthNormalDPT5.get_bins = get_bins
    decoder.RAFTDepthNormalDPT5._blindspot_patched = True


class Metric3D(DepthModel):
    """Metric3D v2 with the preprocessing of the official ``hubconf.py``.

    The image is resized to fit 616x1064 (keeping the aspect ratio) and padded; the focal length is
    scaled by the same factor. The network predicts depth for a canonical camera with focal 1000 px,
    so metric depth is ``pred * fx_scaled / 1000``.

    Args:
        info: Static model description.
        hub_model: Hub entry point, e.g. "metric3d_vit_small".
    """

    def __init__(self, info: ModelInfo, hub_model: str) -> None:
        """Stores the configuration; weights are loaded in :meth:`load`."""
        super().__init__(info)
        self.hub_model = hub_model

    def load(self, device: str) -> None:
        """Loads the hub model (pinned commit) and applies the compatibility fixes."""
        import torch

        self.device = device
        _stub_mmcv()
        model = torch.hub.load(HUB_REPO, self.hub_model, pretrain=True, trust_repo=True)
        _patch_hub_code()
        self.module = model.to(device).eval()

    def predict(self, rgb: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
        """Predicts metric depth in meters at the input resolution."""
        import torch

        h, w = rgb.shape[:2]
        scale = min(INPUT_SIZE[0] / h, INPUT_SIZE[1] / w)
        rh, rw = int(h * scale), int(w * scale)
        k = scaled_intrinsics(intrinsics, (h, w), (rh, rw))
        img = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
        pad_h, pad_w = INPUT_SIZE[0] - rh, INPUT_SIZE[1] - rw
        top, left = pad_h // 2, pad_w // 2
        img = cv2.copyMakeBorder(
            img, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT, value=MEAN.tolist()
        )
        x = torch.from_numpy(((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1))[None].to(self.device)
        with torch.inference_mode():
            pred, _, _ = self.module.inference({"input": x})
        pred = pred[0, 0, top : top + rh, left : left + rw].float().cpu().numpy()
        depth = np.clip(pred * (k.fx / CANONICAL_FOCAL), 0, 300)
        return resize_depth(depth, (h, w))
