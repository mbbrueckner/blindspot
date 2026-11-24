"""Compatibility module: camera calibration now lives in :mod:`blindspot.data.calibration`.

Kept so that existing imports (``from blindspot.calibration import ...``) keep working.
"""

from blindspot.data.calibration import (
    CALIBRATION_FILE,
    calibrate_from_corners,
    find_corners,
    load_calibration,
    save_calibration,
    scale_intrinsics,
)

__all__ = [
    "CALIBRATION_FILE",
    "calibrate_from_corners",
    "find_corners",
    "load_calibration",
    "save_calibration",
    "scale_intrinsics",
]
