"""Compatibility module: the split helpers now live in :mod:`blindspot.data.splits`.

Kept so that existing imports (``from blindspot.splits import ...``) keep working.
"""

from blindspot.data.splits import make_splits, write_splits

__all__ = ["make_splits", "write_splits"]
