"""Models from Hugging Face ``transformers``: Depth Anything V2 (relative and metric), ZoeDepth, Depth Pro."""

import inspect

import numpy as np

from blindspot.core import Intrinsics
from blindspot.models.base import DepthModel, ModelInfo, disparity_to_pseudo_depth, resize_depth


class HFDepth(DepthModel):
    """Any ``AutoModelForDepthEstimation`` checkpoint with its own image processor.

    Args:
        info: Static model description.
        model_id: Hugging Face model ID.
        output: "depth" if the model predicts meters, "disparity" if it predicts relative disparity.
    """

    def __init__(self, info: ModelInfo, model_id: str, output: str) -> None:
        """Stores the configuration; weights are loaded in :meth:`load`."""
        super().__init__(info)
        if output not in ("depth", "disparity"):
            raise ValueError(f"output must be 'depth' or 'disparity', got {output!r}")
        self.model_id = model_id
        self.output = output
        self.processor = None

    def load(self, device: str) -> None:
        """Downloads (once) and loads processor and model."""
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.module = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(device).eval()
        params = inspect.signature(self.processor.post_process_depth_estimation).parameters
        self._needs_source_sizes = "source_sizes" in params

    def predict(self, rgb: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
        """Runs the processor (resize, normalize), the model and the official post-processing."""
        import torch

        h, w = rgb.shape[:2]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.module(**inputs)
        kwargs = {"target_sizes": [(h, w)]}
        if self._needs_source_sizes:
            kwargs["source_sizes"] = [(h, w)]
        pred = self.processor.post_process_depth_estimation(outputs, **kwargs)[0]["predicted_depth"]
        pred = resize_depth(pred.float().cpu().numpy(), (h, w))
        return pred if self.output == "depth" else disparity_to_pseudo_depth(pred)


class DepthProHF(DepthModel):
    """Apple Depth Pro via ``transformers``, using our focal length instead of the predicted field of view.

    The model predicts a canonical inverse depth ``c``; the official post-processing turns it into meters
    with ``depth = f_px / (width * c)``, where ``f_px`` comes from the predicted field of view. We use the
    known ``fx`` instead (the ratio ``fx / width`` does not depend on the resolution).

    Args:
        info: Static model description.
        model_id: Hugging Face model ID.
    """

    def __init__(self, info: ModelInfo, model_id: str) -> None:
        """Stores the configuration; weights are loaded in :meth:`load`."""
        super().__init__(info)
        self.model_id = model_id
        self.processor = None

    def load(self, device: str) -> None:
        """Downloads (once) and loads processor and model."""
        from transformers import AutoImageProcessor, DepthProForDepthEstimation

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.module = DepthProForDepthEstimation.from_pretrained(self.model_id).to(device).eval()

    def predict(self, rgb: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
        """Predicts canonical inverse depth and converts it with our focal length."""
        import torch

        h, w = rgb.shape[:2]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.module(**inputs)
        inv = outputs.predicted_depth.float()
        inv = torch.nn.functional.interpolate(inv[:, None], size=(h, w), mode="bilinear", align_corners=False)
        inv = inv[0, 0].cpu().numpy()
        with np.errstate(divide="ignore"):
            depth = (intrinsics.fx / w) / np.clip(inv, 1e-4, 1e4)
        return depth.astype(np.float32)
