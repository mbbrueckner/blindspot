"""Writes the results of a run as JSON and Markdown.

The Markdown report is built as a list of blocks (headings, paragraphs, tables, figures) and rendered at
the end; figures are linked from ``figures/``.
"""

import json
import math
from pathlib import Path
from typing import Optional, Sequence

from blindspot.eval.metrics import DEPTH_RANGES

ALIGNMENT_LABELS = {
    "raw": "raw",
    "median": "median (oracle)",
    "affine": "affine (oracle)",
}

NOTES = [
    "**Alignment.** `raw` is the prediction as it is (metric models only). `median` (one scale per image) "
    "and `affine` (scale and shift in disparity, least squares per image) use the ground truth: they are an "
    "**oracle upper bound**, not what the system can do at run time. They make relative models (MiDaS, "
    "DA2, ZipDepth) comparable with metric ones. Headline numbers use `raw` for metric and `affine` for "
    "relative models (they predict disparity up to scale and shift, so one scale factor cannot fit them).",
    "**Metrics.** Mean over frames of AbsRel, RMSE (m) and δ1 (share of pixels with max(p/g, g/p) < 1.25), "
    "per reference depth range (0-2, 2-5, 5-10 m and all up to 10 m). Frames without valid pixels in a range "
    "are left out of that range. After alignment, predictions are clipped to [0.001 m, 10 m] (standard "
    "NYU/KITTI protocol), so a few pixels predicted hundreds of meters away cannot dominate a frame.",
    "**Latency** is end to end per frame (pre-processing, network, post-processing to a NumPy array), after "
    "warm-up runs that are not timed; the device is synchronized before the timer stops. Peak memory is the "
    "exact CUDA peak of tensor memory, a sampled value on MPS and not available on the CPU (process RSS is "
    "in the JSON). Load time includes downloading the weights if they were not cached yet. Performance "
    "numbers come from the run that filled the prediction cache (see the device column); rerun with "
    "`--recompute` for fresh timings.",
    "**iPhone ground truth** is LiDAR (ARKit, 256x192, only confidence 2 kept). It is unreliable on glass "
    "and other reflective surfaces, at night and beyond about 5 m; treat the 5-10 m range and the glass, "
    "night and backlight tags with care.",
    "**SANPO ground truth** is dense stereo depth (CREStereo) that starts at about 2 m, so the 0-2 m range "
    "is empty there; the near range is only measured on our own recordings.",
]


def fmt(value, digits: int = 3) -> str:
    """Formats a number for a table ("–" for missing values)."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


class Report:
    """Collects report blocks and renders them as Markdown.

    Args:
        title: Document title.
    """

    def __init__(self, title: str) -> None:
        """Starts an empty report."""
        self.title = title
        self.blocks: list[tuple] = []

    def h(self, level: int, text: str) -> None:
        """Adds a heading (level 2 or 3)."""
        self.blocks.append(("h", level, text))

    def p(self, text: str) -> None:
        """Adds a paragraph (inline Markdown for bold and code is supported)."""
        self.blocks.append(("p", text))

    def items(self, items: Sequence[str]) -> None:
        """Adds a bullet list."""
        self.blocks.append(("ul", list(items)))

    def table(self, headers: Sequence[str], rows: Sequence[Sequence]) -> None:
        """Adds a table; cells are formatted with :func:`fmt`."""
        self.blocks.append(("table", list(headers), [[fmt(c) for c in r] for r in rows]))

    def figure(self, path: Path, caption: str) -> None:
        """Adds a figure (path relative to the report folder)."""
        self.blocks.append(("fig", Path(path), caption))

    def markdown(self) -> str:
        """Renders the report as Markdown."""
        out = [f"# {self.title}", ""]
        for b in self.blocks:
            if b[0] == "h":
                out += ["#" * b[1] + " " + b[2], ""]
            elif b[0] == "p":
                out += [b[1], ""]
            elif b[0] == "ul":
                out += [f"- {i}" for i in b[1]] + [""]
            elif b[0] == "table":
                out += ["| " + " | ".join(b[1]) + " |", "|" + "---|" * len(b[1])]
                out += ["| " + " | ".join(r) + " |" for r in b[2]] + [""]
            elif b[0] == "fig":
                out += [f"![{b[2]}]({b[1].as_posix()})", f"*{b[2]}*", ""]
        return "\n".join(out)


def range_table_rows(summary, model_names, dataset, alignment_of) -> list[list]:
    """Rows "model, alignment, AbsRel per range, δ1 per range" for one dataset."""
    rows = []
    for name in model_names:
        mode = alignment_of(name)
        cells = {
            r["range"]: r
            for r in summary
            if r["model"] == name and r["dataset"] == dataset and r["alignment"] == mode
        }
        if not cells:
            continue
        rows.append(
            [name, ALIGNMENT_LABELS[mode]]
            + [cells.get(rng, {}).get("abs_rel") for rng in DEPTH_RANGES]
            + [cells.get(rng, {}).get("delta1") for rng in DEPTH_RANGES]
            + [cells.get("all", {}).get("rmse"), cells.get("all", {}).get("n_frames")]
        )
    return rows


def write_json(path: Path, data: dict) -> None:
    """Writes JSON with NaN as null (valid JSON for other tools)."""

    def clean(o):
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, Path):
            return str(o)
        return o

    path.write_text(json.dumps(clean(data), indent=1) + "\n")


def perf_row(name: str, status: dict) -> list:
    """Row of the model overview table."""
    perf: Optional[dict] = status.get("perf") or {}
    info = status["info"]
    return [
        name,
        info["label"],
        info["kind"],
        "yes" if info["uses_intrinsics"] else "no",
        status["status"],
        None if perf.get("n_params") is None else perf["n_params"] / 1e6,
        perf.get("load_s"),
        perf.get("mean_ms"),
        perf.get("p95_ms"),
        perf.get("fps"),
        perf.get("peak_mem_mb"),
        perf.get("device"),
    ]


PERF_HEADERS = [
    "model",
    "name",
    "kind",
    "uses K",
    "status",
    "params [M]",
    "load [s]",
    "mean [ms]",
    "p95 [ms]",
    "FPS",
    "peak mem [MiB]",
    "device",
]
