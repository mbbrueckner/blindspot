"""Registry of all depth models: short names, groups and lazy construction.

The registry only holds :class:`~blindspot.models.base.ModelInfo` and an import path, so listing models
needs no torch. :func:`create` imports the wrapper module when a model is actually used.
"""

import importlib
import os
from pathlib import Path

from blindspot.models.base import DepthModel, ModelInfo

_HF = "extra: models"
_GITHUB_ZIPDEPTH = "extra: github (zipdepth @ git+https://github.com/fabiotosi92/ZipDepth)"

# name -> (info, "module:Class", constructor kwargs)
_SPECS: dict[str, tuple[ModelInfo, str, dict]] = {}


def _register(info: ModelInfo, target: str, **kwargs) -> None:
    _SPECS[info.name] = (info, target, kwargs)


# ---- Tier 1 ----
_register(
    ModelInfo(
        "midas_s",
        "MiDaS v2.1 Small",
        "MiDaS",
        "relative",
        False,
        "torch.hub intel-isl/MiDaS:v3_1 MiDaS_small",
        notes="input 256 px (long side), outputs relative disparity",
    ),
    "blindspot.models.midas:MiDaS",
    hub_model="MiDaS_small",
)
_register(
    ModelInfo(
        "da2_s",
        "Depth Anything V2 Small",
        "Depth Anything V2",
        "relative",
        False,
        "depth-anything/Depth-Anything-V2-Small-hf",
        notes="input 518 px (short side), relative disparity",
    ),
    "blindspot.models.hf:HFDepth",
    model_id="depth-anything/Depth-Anything-V2-Small-hf",
    output="disparity",
)
_register(
    ModelInfo(
        "da2m_in_s",
        "Depth Anything V2 Metric Indoor Small",
        "Depth Anything V2",
        "metric",
        False,
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        notes="fine-tuned on Hypersim, max 20 m",
    ),
    "blindspot.models.hf:HFDepth",
    model_id="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    output="depth",
)
_register(
    ModelInfo(
        "da2m_out_s",
        "Depth Anything V2 Metric Outdoor Small",
        "Depth Anything V2",
        "metric",
        False,
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        notes="fine-tuned on Virtual KITTI 2, max 80 m",
    ),
    "blindspot.models.hf:HFDepth",
    model_id="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    output="depth",
)
_register(
    ModelInfo(
        "m3d_s",
        "Metric3D v2 ViT-S",
        "Metric3D v2",
        "metric",
        True,
        "torch.hub yvanyin/metric3d metric3d_vit_small",
        install="extra: models (needs mmengine, no mmcv)",
        notes="input 616x1064, canonical focal 1000 px",
    ),
    "blindspot.models.metric3d:Metric3D",
    hub_model="metric3d_vit_small",
)
_register(
    ModelInfo(
        "yolo26n",
        "YOLO26n-Depth",
        "YOLO26 Depth",
        "metric",
        False,
        "ultralytics yolo26n-depth.pt",
        notes="input 768 px",
    ),
    "blindspot.models.yolo:YOLODepth",
    weights="yolo26n-depth.pt",
)
_register(
    ModelInfo(
        "zipdepth",
        "ZipDepth Base",
        "ZipDepth",
        "relative",
        False,
        "github fabiotosi92/ZipDepth checkpoints/zipdepth_base.pth",
        install=_GITHUB_ZIPDEPTH,
        notes="input 384 px (short side), relative disparity",
    ),
    "blindspot.models.zipdepth:ZipDepth",
)

# ---- Tier 2 ----
_register(
    ModelInfo(
        "moge2_s",
        "MoGe-2 ViT-S",
        "MoGe-2",
        "metric",
        True,
        "Ruicheng/moge-2-vits-normal",
        install="separate env, see docs/models.md",
        env="moge",
    ),
    "blindspot.models.moge:MoGe2",
    model_id="Ruicheng/moge-2-vits-normal",
)
_register(
    ModelInfo(
        "unidepth2_s",
        "UniDepthV2 ViT-S",
        "UniDepth",
        "metric",
        True,
        "lpiccinelli/unidepth-v2-vits14",
        install="separate env, see docs/models.md",
        env="unidepth",
    ),
    "blindspot.models.unidepth:UniDepth2",
    model_id="lpiccinelli/unidepth-v2-vits14",
)
_register(
    ModelInfo(
        "depthpro",
        "Depth Pro",
        "Depth Pro",
        "metric",
        True,
        "apple/DepthPro-hf",
        notes="input 1536x1536, focal from our intrinsics instead of the predicted FOV",
    ),
    "blindspot.models.hf:DepthProHF",
    model_id="apple/DepthPro-hf",
)
_register(
    ModelInfo(
        "da3m_l",
        "Depth Anything 3 Metric Large",
        "Depth Anything 3",
        "metric",
        True,
        "depth-anything/DA3METRIC-LARGE",
        install="separate env, see docs/models.md",
        env="da3",
        notes="metric depth = focal * output / 300",
    ),
    "blindspot.models.da3:DA3",
    model_id="depth-anything/DA3METRIC-LARGE",
    metric=True,
)
_register(
    ModelInfo(
        "da3_s",
        "Depth Anything 3 Small",
        "Depth Anything 3",
        "relative",
        False,
        "depth-anything/DA3-SMALL",
        install="separate env, see docs/models.md",
        env="da3",
        notes="relative depth (not disparity)",
    ),
    "blindspot.models.da3:DA3",
    model_id="depth-anything/DA3-SMALL",
    metric=False,
)
_register(
    ModelInfo(
        "zoedepth",
        "ZoeDepth NYU+KITTI",
        "ZoeDepth",
        "metric",
        False,
        "Intel/zoedepth-nyu-kitti",
        notes="routes indoor/outdoor heads automatically",
    ),
    "blindspot.models.hf:HFDepth",
    model_id="Intel/zoedepth-nyu-kitti",
    output="depth",
)
_register(
    ModelInfo(
        "m3d_l",
        "Metric3D v2 ViT-L",
        "Metric3D v2",
        "metric",
        True,
        "torch.hub yvanyin/metric3d metric3d_vit_large",
        install="extra: models (needs mmengine, no mmcv)",
        notes="input 616x1064, canonical focal 1000 px",
    ),
    "blindspot.models.metric3d:Metric3D",
    hub_model="metric3d_vit_large",
)

# ---- Baselines ----
_register(
    ModelInfo(
        "constant",
        "Constant 6 m",
        "Baseline",
        "metric",
        False,
        "none",
        install="none",
        notes="lower bound for every metric: the same depth everywhere, so it never alerts within 5 m",
    ),
    "blindspot.models.constant:ConstantDepth",
    depth_m=6.0,
)

TIER1 = ("midas_s", "da2_s", "da2m_in_s", "da2m_out_s", "m3d_s", "yolo26n", "zipdepth")
BASELINES = ("constant",)
TIER2 = ("moge2_s", "unidepth2_s", "depthpro", "da3m_l", "da3_s", "zoedepth", "m3d_l")

GROUPS: dict[str, tuple[str, ...]] = {
    "tier1": TIER1,
    "tier2": TIER2,
    "baseline": BASELINES,
    # small enough to be candidates for the phone (< 30 M parameters)
    "mobile": ("midas_s", "yolo26n", "zipdepth", "da2_s", "da2m_in_s", "da2m_out_s"),
    # use the focal length we pass in
    "camera": tuple(n for n, (i, _, _) in _SPECS.items() if i.uses_intrinsics),
    # large models as an accuracy reference
    "reference": ("depthpro", "da3m_l", "m3d_l"),
    "metric": tuple(n for n, (i, _, _) in _SPECS.items() if i.kind == "metric"),
    "relative": tuple(n for n, (i, _, _) in _SPECS.items() if i.kind == "relative"),
    # runnable in the project environment (no separate env)
    "main": tuple(n for n, (i, _, _) in _SPECS.items() if i.env == "main"),
    "all": TIER1 + TIER2 + BASELINES,
}


def names() -> list[str]:
    """All registered model names in registry order."""
    return list(_SPECS)


def info(name: str) -> ModelInfo:
    """Static info of a model.

    Raises:
        KeyError: If the name is unknown (the message lists the known names).
    """
    if name not in _SPECS:
        raise KeyError(f"Unknown model {name!r}. Known: {', '.join(_SPECS)}")
    return _SPECS[name][0]


def resolve(selection: str | list[str]) -> list[str]:
    """Expands group names and model names into a list of unique model names.

    Args:
        selection: Comma-separated string or list of model and group names, e.g. "tier1,depthpro".

    Returns:
        Model names in the given order, without duplicates.

    Raises:
        KeyError: If an entry is neither a model nor a group.
    """
    items = selection.split(",") if isinstance(selection, str) else selection
    out: list[str] = []
    for item in (i.strip() for i in items):
        if not item:
            continue
        if item in GROUPS:
            out += GROUPS[item]
        elif item in _SPECS:
            out.append(item)
        else:
            raise KeyError(
                f"{item!r} is neither a model nor a group. Models: {', '.join(_SPECS)}; "
                f"groups: {', '.join(GROUPS)}"
            )
    return list(dict.fromkeys(out))


def create(name: str) -> DepthModel:
    """Instantiates a model wrapper (imports its module; does not load weights yet).

    Raises:
        KeyError: If the name is unknown.
        ImportError: If the wrapper's dependencies are missing.
    """
    model_info, target, kwargs = _SPECS[info(name).name]
    module_name, cls_name = target.split(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(model_info, **kwargs)


def register(model_info: ModelInfo, target: str, **kwargs) -> None:
    """Adds a model at run time (used by tests for dummy models).

    Args:
        model_info: Static info; its ``name`` must be new.
        target: Import path ``"module:Class"``; the class takes ``(info, **kwargs)``.
        **kwargs: Constructor arguments.

    Raises:
        ValueError: If the name is taken.
    """
    if model_info.name in _SPECS:
        raise ValueError(f"Model {model_info.name!r} is already registered")
    _register(model_info, target, **kwargs)


def unregister(name: str) -> None:
    """Removes a model added with :func:`register`."""
    _SPECS.pop(name, None)


def weights_dir() -> Path:
    """Directory for downloaded weights: ``$BLINDSPOT_WEIGHTS`` or ``~/.cache/blindspot/weights``."""
    path = Path(os.environ.get("BLINDSPOT_WEIGHTS", Path.home() / ".cache" / "blindspot" / "weights"))
    path.mkdir(parents=True, exist_ok=True)
    return path
