"""Compatibility module: the SANPO loader now lives in :mod:`blindspot.data.sanpo`.

Kept so that existing imports (``from blindspot.sanpo import ...``) keep working.
"""

from blindspot.data.sanpo import (
    CHEST,
    SANPO_URL,
    frame_indices,
    load_sample,
    read_depth,
    read_intrinsics,
    read_session_ids,
)

__all__ = [
    "CHEST",
    "SANPO_URL",
    "frame_indices",
    "load_sample",
    "read_depth",
    "read_intrinsics",
    "read_session_ids",
]
