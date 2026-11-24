"""Shared look of all figures: colors per model group, short model names, matplotlib setup, saving.

Models are colored by how they get to meters (see docs/models.md): metric without camera information,
metric with focal length, relative. The three colors are the first three slots of a validated categorical
palette (distinguishable with color vision deficiencies). Every point or line also carries the model name,
so color is never the only cue.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

GROUPS = ("focal", "metric", "relative")
GROUP_COLORS = {"focal": "#2a78d6", "metric": "#eb6834", "relative": "#1baf7a"}
GROUP_LABELS = {
    "focal": "metric, uses focal length",
    "metric": "metric, no camera info",
    "relative": "relative (needs scaling)",
}

DATASET_LABELS = {"iphone": "own recordings", "sanpo": "SANPO"}
TAG_VALUE_LABELS = {"in": "indoor", "out": "outdoor", "trans": "transition"}

SHORT_NAMES = {
    "midas_s": "MiDaS S",
    "da2_s": "DA2 S",
    "da2m_in_s": "DA2 Metric Indoor S",
    "da2m_out_s": "DA2 Metric Outdoor S",
    "m3d_s": "Metric3D S",
    "yolo26n": "YOLO26n",
    "zipdepth": "ZipDepth",
    "moge2_s": "MoGe-2 S",
    "unidepth2_s": "UniDepthV2 S",
    "depthpro": "Depth Pro",
    "da3m_l": "DA3 Metric L",
    "da3_s": "DA3 S",
    "zoedepth": "ZoeDepth",
    "m3d_l": "Metric3D L",
}


def group_of(info: dict) -> str:
    """Model group: "focal", "metric" or "relative" (from the registry info dict)."""
    if info.get("kind") == "relative":
        return "relative"
    return "focal" if info.get("uses_intrinsics") else "metric"


def dataset_label(dataset: str) -> str:
    """Display name of a dataset."""
    return DATASET_LABELS.get(dataset, dataset)


def bottom_legend(ax, handles: list, ncol: int = 3) -> None:
    """Legend centered below the axes, so it never covers data or the title."""
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=ncol, fontsize=9)


def short_name(name: str) -> str:
    """Display name of a model (the registry name if there is none)."""
    return SHORT_NAMES.get(name, name)


def setup() -> None:
    """Applies a clean, presentation-ready style (recessive axes and grid)."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "font.family": "sans-serif",
            "lines.linewidth": 2,
            "pdf.fonttype": 42,  # embed TrueType fonts, so text in the PDF stays selectable
        }
    )


def plain_log_axis(axis) -> None:
    """Plain number labels (2, 5, 10, 20, ...) on a log-scaled axis."""
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def save(fig, stem: Path) -> Path:
    """Saves a figure as ``<stem>.png`` (200 dpi, for slides) and ``<stem>.pdf`` (vector, for LaTeX).

    Returns:
        Path of the PNG.
    """
    stem = Path(stem)
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")
