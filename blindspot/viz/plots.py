"""Performance figures of an evaluation run, for the report and for slides.

"Scaled" always means one scale factor per frame from the reference depth (median alignment). It is an
oracle, but it is what ground scaling (WP 2.2) will try to deliver, and it treats every model the same way.
Every function writes ``<stem>.png`` and ``<stem>.pdf`` and returns False if there is nothing to plot.
"""

import math
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from blindspot.eval.metrics import DEPTH_RANGES
from blindspot.viz import style

SCALED = "median"
SCALED_LABEL = "δ1 with one scale factor per frame"


def _lookup(summary: Sequence[dict]) -> dict:
    return {(r["model"], r["dataset"], r["alignment"], r["range"]): r for r in summary}


def _value(table: dict, model: str, dataset: str, alignment: str, rng: str = "all", key: str = "delta1"):
    row = table.get((model, dataset, alignment, rng))
    if row is None or row["n_frames"] == 0 or row[key] is None or not math.isfinite(row[key]):
        return None
    return row[key]


def _group_legend(groups: set) -> list:
    return [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=8,
            color=style.GROUP_COLORS[g],
            label=style.GROUP_LABELS[g],
        )
        for g in style.GROUPS
        if g in groups
    ]


def _place_labels(ax, points: list[tuple[float, float, str]], fontsize: int = 8) -> None:
    """Writes a label next to every point, trying several offsets so labels do not overlap."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    taken = []
    offsets = [
        (7, 3),
        (7, -11),
        (-7, 3),
        (-7, -11),
        (7, 12),
        (7, -20),
        (-7, 12),
        (-7, -20),
        (0, 14),
        (0, -22),
    ]
    for x, y, text in sorted(points, key=lambda p: -p[1]):
        for dx, dy in offsets:
            t = ax.annotate(
                text,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=fontsize,
                color=style.INK_SECONDARY,
                ha="left" if dx >= 0 else "right",
            )
            box = t.get_window_extent(renderer).expanded(1.05, 1.15)
            if not any(box.overlaps(o) for o in taken):
                taken.append(box)
                break
            t.remove()
        else:  # no free spot: keep the first position
            ax.annotate(
                text,
                (x, y),
                xytext=offsets[0],
                textcoords="offset points",
                fontsize=fontsize,
                color=style.INK_SECONDARY,
            )


def accuracy_vs_latency(summary, infos: dict, perf: dict, dataset: str, stem: Path) -> bool:
    """Scaled δ1 over latency; point size = parameters, color = model group, dashed line = Pareto front.

    Args:
        summary: Aggregated rows from :func:`blindspot.eval.scoring.aggregate`.
        infos: Model name -> registry info dict.
        perf: Model name -> performance dict (``mean_ms``, ``n_params``).
        dataset: Dataset to plot.
        stem: Output path without extension.
    """
    table = _lookup(summary)
    pts = []
    for name, info in infos.items():
        p = perf.get(name) or {}
        y = _value(table, name, dataset, SCALED)
        if y is None or not p.get("mean_ms"):
            continue
        pts.append((p["mean_ms"], y, name, style.group_of(info), (p.get("n_params") or 0) / 1e6))
    if not pts:
        return False
    style.setup()
    fig, ax = plt.subplots(figsize=(8, 5))
    for x, y, name, group, params in pts:
        ax.scatter(
            x,
            y,
            s=30 + 12 * math.sqrt(params),
            color=style.GROUP_COLORS[group],
            alpha=0.9,
            edgecolors=style.SURFACE,
            linewidths=1.5,
            zorder=3,
        )
    front, best = [], -1.0
    for x, y, *_ in sorted(pts):
        if y > best:
            front.append((x, y))
            best = y
    if len(front) > 1:
        fx, fy = zip(*front)
        ax.step(fx, fy, where="post", color=style.INK_MUTED, linestyle="--", linewidth=1.2, zorder=2)
    ax.set_xscale("log")
    style.plain_log_axis(ax.xaxis)
    ys = [p[1] for p in pts]
    ax.set_ylim(max(0.0, min(ys) - 0.08), min(1.0, max(ys) + 0.05))
    xs = [p[0] for p in pts]
    ax.set_xlim(min(xs) / 1.6, max(xs) * 1.6)
    _place_labels(ax, [(x, y, style.short_name(n)) for x, y, n, *_ in pts])
    ax.set_xlabel("latency per frame on the GPU [ms, log scale]")
    ax.set_ylabel(f"{SCALED_LABEL} (higher is better)")
    ax.set_title(f"Accuracy vs. speed ({style.dataset_label(dataset)})", loc="left")
    handles = _group_legend({p[3] for p in pts})
    handles += [
        Line2D([], [], color=style.INK_MUTED, linestyle="--", linewidth=1.2, label="best trade-off (Pareto)")
    ]
    handles += [
        plt.scatter(
            [], [], s=30 + 12 * math.sqrt(m), color=style.INK_MUTED, alpha=0.5, label=f"{m} M parameters"
        )
        for m in (10, 100, 1000)
    ]
    ax.legend(handles=handles, loc="lower right", labelspacing=0.9)
    style.save(fig, stem)
    return True


def raw_vs_scaled(summary, infos: dict, dataset: str, stem: Path) -> bool:
    """Per model: δ1 of the raw metric output and after one scale factor per frame (dumbbell chart).

    Relative models have no raw value; they are shown with the scaled value only.
    """
    table = _lookup(summary)
    rows = []
    for name, info in infos.items():
        scaled = _value(table, name, dataset, SCALED)
        if scaled is not None:
            rows.append((scaled, _value(table, name, dataset, "raw"), name, style.group_of(info)))
    if not rows:
        return False
    rows.sort(key=lambda r: (r[0], r[2]))
    style.setup()
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.4))
    for i, (scaled, raw, name, group) in enumerate(rows):
        color = style.GROUP_COLORS[group]
        if raw is not None:
            ax.plot([raw, scaled], [i, i], color=color, alpha=0.45, linewidth=3, solid_capstyle="round")
            ax.scatter(raw, i, s=60, color=style.SURFACE, edgecolors=color, linewidths=2, zorder=3)
            ax.text(raw - 0.012, i, f"{raw:.2f}", ha="right", va="center", fontsize=8, color=style.INK_MUTED)
        ax.scatter(scaled, i, s=70, color=color, zorder=3)
        ax.text(
            scaled + 0.012, i, f"{scaled:.2f}", ha="left", va="center", fontsize=8, color=style.INK_SECONDARY
        )
    ax.set_yticks(range(len(rows)), [style.short_name(r[2]) for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.04, 1.06)
    ax.set_xlabel("δ1 (share of pixels within 25 % of the true depth)")
    ax.set_title(f"Raw output vs. one scale factor per frame ({style.dataset_label(dataset)})", loc="left")
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=style.SURFACE,
            markeredgecolor=style.INK_MUTED,
            markeredgewidth=2,
            label="raw (meters as predicted)",
        ),
        Line2D(
            [], [], marker="o", linestyle="", markersize=8, color=style.INK_MUTED, label="with scale factor"
        ),
    ] + _group_legend({r[3] for r in rows})
    style.bottom_legend(ax, handles)
    style.save(fig, stem)
    return True


def near_range(summary, infos: dict, dataset: str, stem: Path, min_share: float = 0.01) -> bool:
    """Scaled δ1 per depth range for every model, sorted by the nearest range.

    Ranges with less than ``min_share`` of the evaluated pixels are left out (e.g. 5-10 m on the iPhone,
    where the LiDAR hardly measures, or 0-2 m on SANPO, whose stereo depth starts at 2 m).
    """
    table = _lookup(summary)
    ranges = []
    for rng in DEPTH_RANGES:
        if rng == "all":
            continue
        pix = sum(
            r["n_pixels"] for k, r in table.items() if k[1] == dataset and k[2] == SCALED and k[3] == rng
        )
        total = sum(
            r["n_pixels"] for k, r in table.items() if k[1] == dataset and k[2] == SCALED and k[3] == "all"
        )
        if total and pix / total >= min_share:
            ranges.append(rng)
    if not ranges:
        return False
    rows = []
    for name, info in infos.items():
        vals = [_value(table, name, dataset, SCALED, rng) for rng in ranges]
        if vals[0] is not None:
            rows.append((vals, name, style.group_of(info)))
    if not rows:
        return False
    rows.sort(key=lambda r: r[0][0])
    markers = ["o", "s", "D"]
    style.setup()
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.4))
    for i, (vals, name, group) in enumerate(rows):
        color = style.GROUP_COLORS[group]
        present = [v for v in vals if v is not None]
        if len(present) > 1:
            ax.plot([min(present), max(present)], [i, i], color=color, alpha=0.35, linewidth=3)
        for j, v in enumerate(vals):
            if v is not None:
                ax.scatter(
                    v,
                    i,
                    s=60,
                    marker=markers[j],
                    zorder=3,
                    color=color if j == 0 else style.SURFACE,
                    edgecolors=color,
                    linewidths=2,
                )
        ax.text(
            vals[0] - 0.012,
            i,
            f"{vals[0]:.2f}",
            ha="right",
            va="center",
            fontsize=8,
            color=style.INK_SECONDARY,
        )
    ax.set_yticks(range(len(rows)), [style.short_name(r[1]) for r in rows])
    ax.grid(axis="y", visible=False)
    lo = min(v for r in rows for v in r[0] if v is not None)
    ax.set_xlim(max(0.0, lo - 0.1), 1.02)
    ax.set_xlabel(f"{SCALED_LABEL}")
    ax.set_title(f"Accuracy by distance ({style.dataset_label(dataset)})", loc="left")
    handles = [
        Line2D(
            [],
            [],
            marker=markers[j],
            linestyle="",
            markersize=8,
            markeredgewidth=2,
            color=style.INK_MUTED,
            markerfacecolor=style.INK_MUTED if j == 0 else style.SURFACE,
            label=f"{rng} m",
        )
        for j, rng in enumerate(ranges)
    ] + _group_legend({r[2] for r in rows})
    style.bottom_legend(ax, handles)
    style.save(fig, stem)
    return True


def error_over_distance(
    curves, infos: dict, dataset: str, highlight: Sequence[str], stem: Path, warn_range_m: float = 3.0
) -> bool:
    """AbsRel over the true distance (0.5 m bins, one scale factor per frame) for a few chosen models."""
    chosen = [
        c for c in curves if c["dataset"] == dataset and c["alignment"] == SCALED and c["model"] in highlight
    ]
    chosen.sort(key=lambda c: list(highlight).index(c["model"]))
    if not chosen:
        return False
    style.setup()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    linestyles = ["-", "--", ":", "-."]
    seen: dict[str, int] = {}
    xmax = 0.0
    for c in chosen:
        info = infos.get(c["model"], {})
        group = style.group_of(info)
        k = seen.get(group, 0)
        seen[group] = k + 1
        x = np.array(c["bin_centers_m"])
        y = np.array([np.nan if v is None else v for v in c["abs_rel"]], dtype=float)
        ok = np.isfinite(y)
        if not ok.any():
            continue
        xmax = max(xmax, float(x[ok].max()))
        ax.plot(
            x[ok],
            y[ok],
            color=style.GROUP_COLORS[group],
            linestyle=linestyles[k % 4],
            marker="o",
            markersize=4,
            label=style.short_name(c["model"]),
        )
    ax.axvspan(0, warn_range_m, color=style.GRID, alpha=0.5, zorder=0)
    ax.text(
        0.1,
        0.97,
        f"alert range (< {warn_range_m:g} m)",
        transform=ax.get_xaxis_transform(),
        fontsize=8,
        color=style.INK_MUTED,
        va="top",
    )
    ax.set_xlim(0, xmax + 0.25)
    ax.set_ylim(0, None)
    ax.set_xlabel("true distance [m]")
    ax.set_ylabel("AbsRel, one scale factor per frame (lower is better)")
    ax.set_title(f"Error over distance ({style.dataset_label(dataset)})", loc="left")
    style.bottom_legend(ax, ax.get_legend_handles_labels()[0])
    style.save(fig, stem)
    return True


def conditions(
    by_tag, infos: dict, dataset: str, stem: Path, pairs: Optional[Sequence[tuple[str, str, str]]] = None
) -> bool:
    """Scaled δ1 per model under two conditions each (default: day vs. night, indoor vs. outdoor).

    Args:
        by_tag: Rows from :func:`blindspot.eval.scoring.aggregate_by_tag`.
        infos: Model name -> registry info dict.
        dataset: Dataset to plot (needs the tags of our own recordings).
        stem: Output path without extension.
        pairs: (tag, value A, value B) per panel.
    """
    pairs = pairs or [("light", "day", "night"), ("env", "in", "out")]
    table = {
        (r["model"], r["tag"], r["value"]): r
        for r in by_tag
        if r["dataset"] == dataset and r["alignment"] == SCALED
    }

    def pair(name, tag, a, b):
        ra, rb = table.get((name, tag, a)), table.get((name, tag, b))
        return (ra, rb) if ra and rb and ra["n_frames"] and rb["n_frames"] else None

    pairs = [p for p in pairs if any(pair(n, *p) for n in infos)]
    names = [n for n in infos if pairs and all(pair(n, *p) for p in pairs)]
    if not names:
        return False
    first = pairs[0]
    names.sort(key=lambda n: pair(n, *first)[0]["delta1"])  # same order in all panels
    style.setup()
    fig, axes = plt.subplots(
        1, len(pairs), figsize=(4.6 * len(pairs) + 1.6, 0.42 * len(names) + 1.8), sharey=True, squeeze=False
    )
    lo = 1.0
    for ax, (tag, a, b) in zip(axes[0], pairs):
        for i, name in enumerate(names):
            ra, rb = pair(name, tag, a, b)
            color = style.GROUP_COLORS[style.group_of(infos[name])]
            ax.plot([ra["delta1"], rb["delta1"]], [i, i], color=color, alpha=0.4, linewidth=3)
            ax.scatter(ra["delta1"], i, s=60, color=color, zorder=3)
            ax.scatter(rb["delta1"], i, s=60, color=style.SURFACE, edgecolors=color, linewidths=2, zorder=3)
            lo = min(lo, ra["delta1"], rb["delta1"])
        ra, rb = pair(names[0], tag, a, b)
        la, lb = (style.TAG_VALUE_LABELS.get(v, v) for v in (a, b))
        ax.set_title(
            f"{la} (●, {ra['n_frames']} frames) vs. {lb} (○, {rb['n_frames']})", loc="left", fontsize=10
        )
        ax.set_xlabel(SCALED_LABEL)
        ax.grid(axis="y", visible=False)
    for ax in axes[0]:
        ax.set_xlim(max(0.0, lo - 0.05), 1.02)
    axes[0][0].set_yticks(range(len(names)), [style.short_name(n) for n in names])
    groups = {style.group_of(infos[n]) for n in names}
    style.bottom_legend(axes[0][0], _group_legend(groups))
    fig.suptitle(
        f"Day vs. night and indoor vs. outdoor ({style.dataset_label(dataset)})",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=style.INK,
    )
    style.save(fig, stem)
    return True
