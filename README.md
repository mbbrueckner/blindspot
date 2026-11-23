# blindspot

Obstacle warnings for blind pedestrians from a single chest-mounted camera and monocular depth estimation
(computer vision lab project, FSU Jena).

## Package structure

```
blindspot/
  core.py          shared data types: Sample, Intrinsics, Obstacle, Alert
  data/            data foundation (WP 1.x), every loader returns Sample objects
    sanpo.py         SANPO loader (WP 1.2)
    calibration.py   camera intrinsics from chessboard photos, intrinsics scaling (WP 1.1)
    annotations.py   obstacle annotation JSON (WP 1.4)
    splits.py        train/val/test splits on session level (WP 1.4)
  eval/            evaluation framework (WP 5.1)
    metrics.py       depth error, alert metrics, latency statistics
    runner.py        runs a variant over test images and collects all metrics
  models/          depth models (WP 2.x): common interface, registry, one module per model family
  alerts/          alert logic (WP 3.x): walking corridor, naive baseline
  viz/             plots, image panels and reports
scripts/           thin command line entry points (download, calibration, ...)
tests/             pytest, no network, no model weights
```

`models/`, `alerts/` and `viz/` are still empty and get filled in WP 2.1 and 3.1.

The old module paths `blindspot.evaluation`, `blindspot.sanpo`, `blindspot.calibration` and
`blindspot.splits` still work: they re-export the functions from the new modules. New code should import
from the new paths, e.g. `from blindspot.data.sanpo import load_sample`.

`data/` and `results/` in the repository root are git-ignored. Datasets, model weights and results are
never committed.
