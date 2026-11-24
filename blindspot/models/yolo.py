"""YOLO26 depth models from Ultralytics (metric depth in meters)."""

import numpy as np

from blindspot.core import Intrinsics
from blindspot.models.base import DepthModel, ModelInfo, resize_depth
from blindspot.models.registry import weights_dir


class YOLODepth(DepthModel):
    """Ultralytics depth model, e.g. ``yolo26n-depth.pt``.

    The weights were trained at ``imgsz=768``, which Ultralytics recommends for prediction.

    Args:
        info: Static model description.
        weights: Checkpoint name; downloaded by Ultralytics into the weights directory on first use.
        imgsz: Inference size in pixels.
    """

    def __init__(self, info: ModelInfo, weights: str, imgsz: int = 768) -> None:
        """Stores the configuration; weights are loaded in :meth:`load`."""
        super().__init__(info)
        self.weights = weights
        self.imgsz = imgsz
        self.yolo = None

    def load(self, device: str) -> None:
        """Loads (and downloads once) the checkpoint."""
        from ultralytics import YOLO

        self.device = device
        self._device_arg = {"cuda": "0", "mps": "mps", "cpu": "cpu"}[device]
        self.yolo = YOLO(str(weights_dir() / self.weights))
        self.yolo.to(device)
        self.module = self.yolo.model

    def predict(self, rgb: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
        """Runs the Ultralytics predictor (expects BGR) and returns the depth map in meters."""
        result = self.yolo.predict(rgb[:, :, ::-1], imgsz=self.imgsz, device=self._device_arg, verbose=False)[
            0
        ]
        return resize_depth(result.depth.data.float().cpu().numpy().squeeze(), rgb.shape[:2])
