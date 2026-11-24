"""Runs the depth model evaluation (WP 2.1) and writes report and figures to results/<run-name>/.

Examples::

    uv run python scripts/run_eval.py --preset smoke
    uv run python scripts/run_eval.py --preset full --run-name m1_full
    uv run python scripts/run_eval.py --preset full --models mobile,m3d_s --datasets iphone --device cpu

Predictions are cached in results/cache, so a second run with the same models only recomputes metrics,
report and figures. On Apple Silicon set ``PYTORCH_ENABLE_MPS_FALLBACK=1``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blindspot.device import DEVICES  # noqa: E402
from blindspot.eval.pipeline import PRESETS, make_config, run  # noqa: E402
from blindspot.models import registry  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    """Parses the arguments and runs the evaluation.

    Args:
        argv: Command line arguments; ``None`` uses ``sys.argv``.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", choices=list(PRESETS), default="smoke")
    ap.add_argument("--run-name", help="output folder below results/ (default: <preset>_<date>)")
    ap.add_argument(
        "--models", help=f"models and groups, comma-separated; groups: {', '.join(registry.GROUPS)}"
    )
    ap.add_argument("--datasets", help="comma-separated: sanpo, iphone")
    ap.add_argument("--device", choices=DEVICES)
    ap.add_argument("--sanpo-root", type=Path)
    ap.add_argument("--sanpo-max-frames", type=int)
    ap.add_argument("--iphone-root", type=Path)
    ap.add_argument(
        "--recordings", help="comma-separated recording names (default: all valid test recordings)"
    )
    ap.add_argument("--split", help="manifest split of the recordings, 'all' for every recording")
    ap.add_argument("--stride", type=int, help="take every n-th iPhone frame")
    ap.add_argument("--max-frames", type=int, help="frames per iPhone recording")
    ap.add_argument("--annotations", type=Path, action="append", help="obstacle annotation JSON (repeatable)")
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--recompute", action="store_true", help="ignore cached predictions")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--list-models", action="store_true", help="print models and groups and exit")
    args = ap.parse_args(argv)

    if args.list_models:
        for name in registry.names():
            i = registry.info(name)
            print(f"{name:12s} {i.kind:8s} env={i.env:9s} {i.label}")
        for group, members in registry.GROUPS.items():
            print(f"group {group}: {', '.join(members)}")
        return

    cfg = make_config(
        args.preset,
        run_name=args.run_name,
        models=args.models,
        datasets=tuple(args.datasets.split(",")) if args.datasets else None,
        device=args.device,
        sanpo_root=args.sanpo_root,
        sanpo_max_frames=args.sanpo_max_frames,
        iphone_root=args.iphone_root,
        iphone_recordings=args.recordings.split(",") if args.recordings else None,
        iphone_split=args.split,
        iphone_stride=args.stride,
        iphone_max_frames=args.max_frames,
        annotations=args.annotations,
        warmup=args.warmup,
        recompute=args.recompute or None,
        figures=False if args.no_figures else None,
    )
    if cfg.iphone_split == "all":
        cfg.iphone_split = None
    run(cfg)


if __name__ == "__main__":
    main()
