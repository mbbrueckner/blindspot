"""Data foundation (WP 1.x): dataset loaders, calibration, annotation and splits.

Every loader returns :class:`~blindspot.core.Sample` objects.

Modules:
    sanpo: SANPO loader (WP 1.2).
    calibration: Camera intrinsics from chessboard photos, intrinsics scaling (WP 1.1).
    annotations: Obstacle annotation JSON (WP 1.4).
    splits: Train, val and test splits on session level (WP 1.4).
"""
