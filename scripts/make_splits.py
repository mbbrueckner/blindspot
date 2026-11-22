"""Writes the versioned train, val and test split files for a downloaded SANPO subset (WP 1.4).

``download_sanpo.py`` leaves ``train_sessions.txt`` and ``test_sessions.txt`` in the data folder.
This script splits the train sessions into train and val (on session level, fixed seed) and writes
``sanpo_<kind>_train.txt``, ``..._val.txt`` and ``..._test.txt`` to ``planning/splits/``.

Example::

    python scripts/make_splits.py --kind real
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blindspot.sanpo import read_session_ids  # noqa: E402
from blindspot.splits import make_splits, write_splits  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    """Parses the arguments, builds the splits and writes the split files.

    Args:
        argv: Command line arguments; ``None`` uses ``sys.argv``.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kind", choices=["real", "synthetic"], required=True)
    ap.add_argument("--data", type=Path, default=Path("data/sanpo"), help="folder given to download_sanpo.py")
    ap.add_argument("--out", type=Path, default=Path("planning/splits"))
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    folder = args.data / args.kind
    lists = {}
    for part in ("train", "test"):
        path = folder / f"{part}_sessions.txt"
        if not path.exists():
            sys.exit(f"{path} is missing, run download_sanpo.py with --split {part} first")
        lists[part] = read_session_ids(path)
    splits = make_splits(lists["train"], lists["test"], args.val_fraction, args.seed)
    write_splits(splits, args.out, f"sanpo_{args.kind}")
    print({part: len(ids) for part, ids in splits.items()}, "->", args.out)


if __name__ == "__main__":
    main()
