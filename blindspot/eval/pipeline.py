"""One evaluation run: load data -> predict (cached) -> score -> warnings -> figures -> report.

Results go to ``results/<run_name>/``: ``report.json`` (everything, including per-frame records),
``report.md`` and ``figures/`` (PNG for slides, PDF for LaTeX). Predictions are cached in
``results/cache`` and shared between runs.
"""

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from blindspot import device as dev
from blindspot.alerts import ALERT_LOGICS, CorridorConfig
from blindspot.core import Sample
from blindspot.data import datasets
from blindspot.data.annotations import apply_annotations, load_annotations
from blindspot.data.sanpo import read_session_ids
from blindspot.eval import report as rep
from blindspot.eval import scoring
from blindspot.eval.alignment import ALIGNMENTS
from blindspot.eval.cache import PredictionCache
from blindspot.eval.runner import predict_models
from blindspot.models import registry


@dataclass
class RunConfig:
    """Everything that defines a run (stored in the report for reproducibility).

    Attributes:
        run_name: Output folder name below ``results_root``.
        models: Comma-separated model and group names, see :func:`blindspot.models.registry.resolve`.
        datasets: "sanpo" and/or "iphone".
        sanpo_root: Folder with the SANPO sessions.
        sanpo_sessions: Session IDs; ``None`` uses ``sanpo_split``, else ``test_sessions.txt`` (or all).
        sanpo_split: Split file (one session ID per line), e.g. ``planning/splits/sanpo_real_test.txt``.
        sanpo_stride: Every n-th downloaded SANPO frame.
        sanpo_max_frames: Limit of SANPO frames.
        iphone_root: Folder with the iPhone recordings.
        iphone_recordings: Recording names; ``None`` uses all valid ones (filtered by split).
        iphone_split: Manifest split to use ("test"); ``None`` for all.
        iphone_max_recordings: Use only the first n recordings.
        iphone_stride: Every n-th iPhone frame.
        iphone_max_frames: Limit of frames per recording.
        iphone_scale: Grid size as a multiple of the LiDAR resolution (3 -> 768x576).
        device: "auto", "cuda", "mps" or "cpu".
        warmup: Untimed predictions per model.
        recompute: Ignore cached predictions.
        max_depth: Largest reference depth that counts (meters).
        annotations: Obstacle annotation JSON files (frame_id -> boxes).
        corridor: Parameters of the alert logic.
        figures: Write figures (needs matplotlib).
        highlight: Models shown in the error-over-distance figure (comma-separated; missing ones are skipped).
        results_root: Parent folder of the run folders.
        cache_root: Prediction cache.
    """

    run_name: str
    models: str = "tier1"
    datasets: tuple[str, ...] = ("sanpo", "iphone")
    sanpo_root: Path = Path("data/sanpo/real")
    sanpo_sessions: Optional[list[str]] = None
    sanpo_split: Optional[Path] = None
    sanpo_stride: int = 1
    sanpo_max_frames: Optional[int] = None
    iphone_root: Path = Path("data/iPhone")
    iphone_recordings: Optional[list[str]] = None
    iphone_split: Optional[str] = "test"
    iphone_max_recordings: Optional[int] = None
    iphone_stride: int = 10
    iphone_max_frames: Optional[int] = None
    iphone_scale: int = 3
    device: str = "auto"
    warmup: int = 3
    recompute: bool = False
    max_depth: float = 10.0
    annotations: list[Path] = field(default_factory=list)
    corridor: CorridorConfig = field(default_factory=CorridorConfig)
    figures: bool = True
    highlight: str = "unidepth2_s,m3d_s,da2m_in_s,da2_s,zipdepth,da3m_l"
    results_root: Path = Path("results")
    cache_root: Path = Path("results/cache")


PRESETS: dict[str, dict] = {
    # 3 SANPO frames and the first own recording (every 30th frame, at most 5): checks the whole chain.
    "smoke": {
        "models": "tier1",
        "sanpo_max_frames": 3,
        "iphone_max_recordings": 1,
        "iphone_stride": 30,
        "iphone_max_frames": 5,
        "warmup": 2,
    },
    # All downloaded SANPO test frames and all own test recordings (every 10th frame), all models.
    "full": {"models": "all", "iphone_stride": 10},
}


def make_config(preset: str, **overrides) -> RunConfig:
    """Builds a run configuration from a preset plus overrides (``None`` values are ignored).

    Raises:
        KeyError: If the preset is unknown.
    """
    if preset not in PRESETS:
        raise KeyError(f"Unknown preset {preset!r}, available: {', '.join(PRESETS)}")
    values = {**PRESETS[preset], **{k: v for k, v in overrides.items() if v is not None}}
    values.setdefault("run_name", f"{preset}_{datetime.now():%Y%m%d_%H%M}")
    return RunConfig(**values)


def _git_state() -> str:
    """Commit hash and whether the working tree has changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
        return f"{commit}{' (uncommitted changes)' if dirty else ''}" or "unknown"
    except OSError:
        return "unknown"


def load_samples(cfg: RunConfig, log: Callable[[str], None]) -> tuple[list[Sample], dict]:
    """Loads all configured datasets; a missing dataset is reported and skipped.

    Returns:
        (samples, data info for the report).
    """
    samples: list[Sample] = []
    info: dict = {}
    if "sanpo" in cfg.datasets:
        try:
            sessions = cfg.sanpo_sessions
            if sessions is None and cfg.sanpo_split is not None:
                sessions = read_session_ids(cfg.sanpo_split)
            s = datasets.load_sanpo(cfg.sanpo_root, sessions, cfg.sanpo_stride, cfg.sanpo_max_frames)
            samples += s
            info["sanpo"] = {"frames": len(s), "sessions": sorted({x.tags["session"] for x in s})}
        except (FileNotFoundError, ValueError) as e:
            info["sanpo"] = {"frames": 0, "error": str(e)}
            log(f"SANPO skipped: {e}")
    if "iphone" in cfg.datasets:
        try:
            s, invalid = datasets.load_iphone(
                cfg.iphone_root,
                cfg.iphone_recordings,
                cfg.iphone_split,
                cfg.iphone_stride,
                cfg.iphone_max_frames,
                cfg.iphone_scale,
                max_recordings=cfg.iphone_max_recordings,
            )
            samples += s
            info["iphone"] = {
                "frames": len(s),
                "recordings": sorted({x.tags["recording"] for x in s}),
                "skipped_invalid_names": invalid,
            }
            if invalid:
                log(f"iPhone: skipped {len(invalid)} folders with invalid names: {', '.join(invalid)}")
        except (FileNotFoundError, ValueError) as e:
            info["iphone"] = {"frames": 0, "error": str(e)}
            log(f"iPhone recordings skipped: {e}")
    n_annotated = 0
    for path in cfg.annotations:
        n_annotated += apply_annotations(samples, load_annotations(str(path)))
    info["annotated_frames"] = n_annotated
    return samples, info


def run(cfg: RunConfig, log: Callable[[str], None] = print) -> Path:
    """Executes a full evaluation run.

    Args:
        cfg: Run configuration.
        log: Progress output.

    Returns:
        The run folder.

    Raises:
        RuntimeError: If no frames could be loaded.
    """
    out = Path(cfg.results_root) / cfg.run_name
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    device = dev.select_device(cfg.device)
    model_names = registry.resolve(cfg.models)
    log(f"Run {cfg.run_name}: device {device} ({dev.device_name(device)}), models {', '.join(model_names)}")

    samples, data_info = load_samples(cfg, log)
    if not samples:
        raise RuntimeError(f"No frames loaded: {data_info}")
    log(
        f"Loaded {len(samples)} frames: "
        + ", ".join(f"{k} {v['frames']}" for k, v in data_info.items() if isinstance(v, dict))
    )

    cache = PredictionCache(cfg.cache_root)
    statuses = predict_models(model_names, samples, cache, device, cfg.warmup, cfg.recompute, log)
    usable = {n: statuses[n]["info"]["kind"] for n in model_names if statuses[n]["status"] != "failed"}
    # Partially predicted models (failed midway) are scored on the frames they have.
    for n in model_names:
        if statuses[n]["status"] == "failed" and statuses[n]["n_predicted"] + statuses[n]["n_cached"] > 0:
            usable[n] = statuses[n]["info"]["kind"]

    log("Scoring ...")
    records, binned = scoring.score_frames(usable, samples, cache, ALIGNMENTS, cfg.max_depth)
    summary = scoring.aggregate(records)
    by_tag = scoring.aggregate_by_tag(records)
    curves = scoring.distance_curves(binned)
    metric_models = [n for n, k in usable.items() if k == "metric"]
    warnings = scoring.score_warnings(metric_models, samples, cache, ALERT_LOGICS, cfg.corridor)
    primary = {n: scoring.primary_alignment(k) for n, k in usable.items()}
    ds_names = sorted({s.tags.get("dataset", "?") for s in samples})
    infos = {n: statuses[n]["info"] for n in usable}
    perf = {n: statuses[n]["perf"] for n in usable}

    figures: list[tuple[Path, str]] = []
    if cfg.figures and usable:
        figures = _make_figures(cfg, summary, by_tag, curves, infos, perf, ds_names, fig_dir, log)

    report = _build_report(
        cfg, device, data_info, statuses, model_names, summary, by_tag, warnings, primary, ds_names, figures
    )
    rep.write_json(
        out / "report.json",
        {
            "config": asdict(cfg),
            "device": device,
            "device_name": dev.device_name(device),
            "git": _git_state(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "date": datetime.now().isoformat(timespec="seconds"),
            "data": data_info,
            "models": statuses,
            "summary": summary,
            "by_tag": by_tag,
            "distance_curves": curves,
            "warnings": warnings,
            "frames": [{k: v for k, v in r.items()} for r in records],
            "figures": [str(p) for p, _ in figures],
        },
    )
    (out / "report.md").write_text(report.markdown())
    log(f"Report: {out / 'report.md'}, figures in {fig_dir}")
    return out


def _make_figures(
    cfg, summary, by_tag, curves, infos, perf, ds_names, fig_dir, log
) -> list[tuple[Path, str]]:
    """Writes the performance figures (PNG and PDF).

    Returns:
        (PNG path relative to the run folder, caption) per written figure.
    """
    try:
        from blindspot.viz import plots
    except ImportError as e:
        log(f"Figures skipped (matplotlib missing: {e}); install with: uv sync --extra viz")
        return []
    fig_dir.mkdir(parents=True, exist_ok=True)
    highlight = [n.strip() for n in cfg.highlight.split(",") if n.strip()]
    figs: list[tuple[Path, str]] = []
    for ds in ds_names:
        jobs = [
            (
                "accuracy_vs_latency",
                "Accuracy vs. speed: δ1 with one scale factor per frame over GPU latency",
                lambda stem: plots.accuracy_vs_latency(summary, infos, perf, ds, stem),
            ),
            (
                "raw_vs_scaled",
                "δ1 of the raw metric output vs. with one scale factor per frame",
                lambda stem: plots.raw_vs_scaled(summary, infos, ds, stem),
            ),
            (
                "near_range",
                "δ1 per depth range (one scale factor per frame)",
                lambda stem: plots.near_range(summary, infos, ds, stem),
            ),
            (
                "error_over_distance",
                "AbsRel over the true distance for selected models",
                lambda stem: plots.error_over_distance(
                    curves, infos, ds, highlight, stem, cfg.corridor.warn_range_m
                ),
            ),
            (
                "conditions",
                "δ1 by day vs. night and indoor vs. outdoor",
                lambda stem: plots.conditions(by_tag, infos, ds, stem),
            ),
        ]
        for name, caption, make in jobs:
            stem = fig_dir / f"{ds}_{name}"
            if make(stem):
                figs.append((stem.with_suffix(".png").relative_to(fig_dir.parent), f"{ds}: {caption}"))
    log(f"{len(figs)} figures (PNG and PDF) in {fig_dir}")
    return figs


def _build_report(
    cfg, device, data_info, statuses, model_names, summary, by_tag, warnings, primary, ds_names, figures
) -> rep.Report:
    """Assembles the report blocks."""
    r = rep.Report(f"Blind Spot depth evaluation: {cfg.run_name}")
    r.p(
        f"Date {datetime.now():%Y-%m-%d %H:%M}, device `{device}` ({dev.device_name(device)}), "
        f"git `{_git_state()}`, "
        f"Python {sys.version.split()[0]}. Models: `{cfg.models}`."
    )
    r.h(2, "Data")
    items = []
    for ds, d in data_info.items():
        if not isinstance(d, dict):
            continue
        if d.get("error"):
            items.append(f"**{ds}**: not loaded ({d['error']})")
            continue
        extra = ", ".join(d.get("sessions", d.get("recordings", [])))
        items.append(f"**{ds}**: {d['frames']} frames ({extra})")
        if d.get("skipped_invalid_names"):
            items.append(f"{ds}: skipped folders with invalid names: {', '.join(d['skipped_invalid_names'])}")
    items.append(f"Annotated frames (obstacles): {data_info.get('annotated_frames', 0)}")
    r.items(items)
    r.h(2, "Models and performance")
    r.table(rep.PERF_HEADERS, [rep.perf_row(n, statuses[n]) for n in model_names])
    failed = [n for n in model_names if statuses[n]["status"] == "failed"]
    if failed:
        r.p("**Failed models** (skipped, the run continued):")
        r.items(
            [
                f"`{n}` at {statuses[n]['stage']}: {statuses[n]['error'][:300]} "
                f"(install: {statuses[n]['info']['install']})"
                for n in failed
            ]
        )
    r.h(2, "Notes")
    r.items(rep.NOTES)
    scored = [n for n in model_names if n in primary]
    for ds in ds_names:
        r.h(2, f"Results: {ds}")
        r.h(3, "Headline (metric: raw, relative: affine)")
        ranges = list(rep.DEPTH_RANGES)
        headers = (
            ["model", "alignment"]
            + [f"AbsRel {x}" for x in ranges]
            + [f"δ1 {x}" for x in ranges]
            + ["RMSE all [m]", "frames"]
        )
        r.table(headers, rep.range_table_rows(summary, scored, ds, lambda n: primary[n]))
        for mode in ALIGNMENTS:
            rows = rep.range_table_rows(
                summary,
                [n for n in scored if mode in scoring.alignments_for(statuses[n]["info"]["kind"])],
                ds,
                lambda n, m=mode: m,
            )
            if rows:
                r.h(3, f"Alignment {rep.ALIGNMENT_LABELS[mode]}")
                r.table(headers, rows)
        tag_rows = [t for t in by_tag if t["dataset"] == ds and t["alignment"] == primary.get(t["model"])]
        tags = sorted({(t["tag"], t["value"]) for t in tag_rows if t["tag"] in scoring.GROUP_TAGS})
        if tags:
            r.h(3, "AbsRel per scene tag (0-10 m, headline alignment)")
            r.table(
                ["model"] + [f"{t}={v}" for t, v in tags],
                [
                    [n]
                    + [
                        next(
                            (
                                t["abs_rel"]
                                for t in tag_rows
                                if t["model"] == n and (t["tag"], t["value"]) == tv
                            ),
                            None,
                        )
                        for tv in tags
                    ]
                    for n in scored
                ],
            )
        ws = [w for w in warnings if w["dataset"] == ds]
        if ws:
            r.h(3, f"Alerts (warning range {cfg.corridor.warn_range_m:g} m, metric models, raw depth)")
            if not any(w["n_annotated"] for w in ws):
                r.p(
                    "No obstacle annotation for this dataset yet: recall, false alarm rate, direction and "
                    "distance "
                    "error are not computable. The alert rate (share of frames with an alert) is shown; "
                    "`reference+...` runs the alert logic on the ground-truth depth."
                )
            r.table(
                [
                    "variant",
                    "alert rate",
                    "recall",
                    "false alarm rate",
                    "direction acc.",
                    "distance MAE [m]",
                    "pos.",
                    "neg.",
                ],
                [
                    [
                        w["variant"],
                        w["alert_rate"],
                        w["recall"],
                        w["false_alarm_rate"],
                        w["direction_accuracy"],
                        w["distance_mae_m"],
                        w["n_positive"],
                        w["n_negative"],
                    ]
                    for w in ws
                ],
            )
        for path, caption in figures:
            if path.name.startswith(f"{ds}_") and path.parent.name == "figures":
                r.figure(path, caption)
    r.h(2, "Corridor configuration")
    r.table(["parameter", "value"], [[k, v] for k, v in cfg.corridor.to_dict().items()])
    return r
