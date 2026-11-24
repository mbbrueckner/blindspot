"""Fills the prediction cache for models that run in a separate environment (MoGe, UniDepth, DA3).

Run it with the Python of that environment (see docs/modelle.md); afterwards ``run_eval.py`` in the main
environment finds the cached predictions and scores them. Example::

    envs/unidepth/bin/python scripts/predict.py --models unidepth2_s --preset full

It loads the same frames as ``run_eval.py`` with the same preset and options.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blindspot import device as dev  # noqa: E402
from blindspot.device import DEVICES  # noqa: E402
from blindspot.eval.cache import PredictionCache  # noqa: E402
from blindspot.eval.pipeline import PRESETS, load_samples, make_config  # noqa: E402
from blindspot.eval.runner import predict_models  # noqa: E402
from blindspot.models import registry  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    """Parses the arguments and predicts all frames that are not cached yet.

    Args:
        argv: Command line arguments; ``None`` uses ``sys.argv``.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", required=True, help="models and groups, comma-separated")
    ap.add_argument("--preset", choices=list(PRESETS), default="full")
    ap.add_argument("--datasets", help="comma-separated: sanpo, iphone")
    ap.add_argument("--device", choices=DEVICES, default="auto")
    ap.add_argument("--recordings", help="comma-separated recording names")
    ap.add_argument("--stride", type=int)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)

    cfg = make_config(
        args.preset,
        run_name="predict",
        datasets=tuple(args.datasets.split(",")) if args.datasets else None,
        iphone_recordings=args.recordings.split(",") if args.recordings else None,
        iphone_stride=args.stride,
        iphone_max_frames=args.max_frames,
    )
    samples, info = load_samples(cfg, print)
    print(f"{len(samples)} frames: {info}")
    device = dev.select_device(args.device)
    status = predict_models(
        registry.resolve(args.models),
        samples,
        PredictionCache(cfg.cache_root),
        device,
        cfg.warmup,
        args.recompute,
    )
    failed = [n for n, s in status.items() if s["status"] == "failed"]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
