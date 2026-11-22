"""Runs one variant over a split and writes all metrics as JSON (WP 5.1).

The variant is given as ``module:function`` (a function ``rgb -> (depth_m, alerts)``). Example::

    python scripts/evaluate.py --root data/sanpo/real --split planning/splits/sanpo_real_test.txt \
        --variant blindspot.variants:constant_depth --annotations planning/annotations/sanpo_real_test.json

Without ``--annotations`` only the depth and latency metrics are meaningful, the alert metrics are NaN.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blindspot.evaluation import apply_annotations, evaluate_variant, load_annotations  # noqa: E402
from blindspot.sanpo import load_samples, read_session_ids  # noqa: E402


def load_variant(spec: str):
    """Imports a variant from a ``module:function`` string.

    Args:
        spec: Import path, e.g. ``blindspot.variants:constant_depth``.

    Returns:
        The variant function.

    Raises:
        ValueError: If ``spec`` has no colon.
    """
    module, sep, name = spec.partition(":")
    if not sep or not name:
        raise ValueError(f"Variant must be 'module:function', got {spec!r}")
    return getattr(importlib.import_module(module), name)


def main(argv: list[str] | None = None) -> None:
    """Parses the arguments, runs the evaluation and writes ``results/<variant>_<split>.json``.

    Args:
        argv: Command line arguments; ``None`` uses ``sys.argv``.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, required=True, help="folder with the session directories")
    ap.add_argument("--split", type=Path, required=True, help="split file, one session ID per line")
    ap.add_argument("--variant", required=True, help="module:function")
    ap.add_argument("--annotations", type=Path, help="obstacle annotation JSON (see load_annotations)")
    ap.add_argument("--warn-range", type=float, default=3.0, help="warning range in meters")
    ap.add_argument("--out", type=Path, help="default: results/<function>_<split>.json")
    args = ap.parse_args(argv)

    variant = load_variant(args.variant)
    samples = load_samples(args.root, read_session_ids(args.split))
    n_annotated = apply_annotations(samples, load_annotations(args.annotations)) if args.annotations else 0
    if not samples:
        sys.exit(f"No frames found for {args.split} in {args.root}")
    result = evaluate_variant(variant, samples, warn_range_m=args.warn_range)
    result |= {
        "variant": args.variant,
        "split": args.split.name,
        "n_annotated": n_annotated,
        "warn_range_m": args.warn_range,
    }
    out = args.out or Path("results") / f"{args.variant.split(':')[1]}_{args.split.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
