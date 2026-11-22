"""Writes the obstacle annotation JSON for a split from SANPO segmentation masks (WP 1.4).

Frames without a mask are left out, so they count as "not annotated". Example::

    python scripts/make_annotations.py --root data/sanpo/real --split planning/splits/sanpo_real_test.txt \
        --out planning/annotations/sanpo_real_test.json
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blindspot.masks import load_obstacles  # noqa: E402
from blindspot.sanpo import load_samples, read_session_ids  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    """Parses the arguments, converts the masks of all frames in the split and writes the JSON.

    Args:
        argv: Command line arguments; ``None`` uses ``sys.argv``.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, required=True, help="folder with the session directories")
    ap.add_argument("--split", type=Path, required=True, help="split file, one session ID per line")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-area", type=int, default=400, help="smallest instance in mask pixels")
    args = ap.parse_args(argv)

    annotations = {}
    for s in load_samples(args.root, read_session_ids(args.split)):
        session, _, index = s.frame_id.split("/")  # "<session>/camera_chest/<index>"
        obstacles = load_obstacles(
            args.root / session, int(index), s.rgb.shape[:2], min_area_px=args.min_area
        )
        if obstacles is not None:
            annotations[s.frame_id] = [asdict(o) for o in obstacles]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(annotations, indent=1) + "\n")
    n_obstacles = sum(len(v) for v in annotations.values())
    print(f"{len(annotations)} annotated frames, {n_obstacles} obstacles -> {args.out}")


if __name__ == "__main__":
    main()
