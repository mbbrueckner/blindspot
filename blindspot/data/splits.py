"""Train, val and test splits (WP 1.4), always on session or walk level.

All images of one session or test walk land in the same split, otherwise nearly identical
neighboring frames end up in both train and test (data leak). The lists are small and versioned.
"""

import random
from pathlib import Path


def make_splits(
    train_ids: list[str], test_ids: list[str], val_fraction: float = 0.15, seed: int = 0
) -> dict[str, list[str]]:
    """Splits the training IDs into train and val; test stays unchanged.

    Args:
        train_ids: IDs (sessions or walks) for training and validation.
        test_ids: IDs for testing; must not occur in ``train_ids``.
        val_fraction: Share of the training IDs used for val (at least one ID if possible).
        seed: Random seed; the same seed gives the same splits.

    Returns:
        Dict with the sorted lists ``train``, ``val`` and ``test``.

    Raises:
        ValueError: If train and test overlap.
    """
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"IDs in both train and test: {sorted(overlap)[:3]}")
    ids = sorted(set(train_ids))
    random.Random(seed).shuffle(ids)
    n_val = min(len(ids) - 1, max(1, round(len(ids) * val_fraction))) if len(ids) > 1 else 0
    return {"train": sorted(ids[n_val:]), "val": sorted(ids[:n_val]), "test": sorted(set(test_ids))}


def write_splits(splits: dict[str, list[str]], out_dir: Path, name: str) -> None:
    """Writes ``<name>_train.txt``, ``<name>_val.txt`` and ``<name>_test.txt`` (one ID per line).

    Args:
        splits: Dict as returned by :func:`make_splits`.
        out_dir: Target directory, created if missing.
        name: File name prefix, e.g. "sanpo".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for part, ids in splits.items():
        (out_dir / f"{name}_{part}.txt").write_text("".join(f"{i}\n" for i in ids))
