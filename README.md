# blindspot

Obstacle warnings for blind pedestrians from a single chest-mounted camera and monocular depth estimation
(computer vision lab project, FSU Jena).

## Package structure

```
blindspot/
  core.py            shared data types: Sample (with tags), Intrinsics, Obstacle, Alert
  device.py          device selection (cuda, mps, cpu), synchronization, peak memory; the only
                     module with GPU-specific code
  data/              data foundation (WP 1.x), every loader returns Sample objects
    sanpo.py           SANPO loader (WP 1.2)
    stray.py           Stray Scanner loader for our iPhone recordings (WP 1.3)
    recordings.py      recording names, validation and manifest (WP 1.3)
    datasets.py        loads SANPO and iPhone data with frame subsampling and tags
    calibration.py     camera intrinsics from chessboard photos, intrinsics scaling (WP 1.1)
    annotations.py     obstacle annotation JSON (WP 1.4)
    splits.py          train/val/test splits on session level (WP 1.4)
  models/            depth models (WP 2.1)
    base.py            common interface DepthModel.predict(rgb, intrinsics) -> depth
    registry.py        short names, groups (tier1, mobile, camera, ...), lazy construction
    midas.py, hf.py (Depth Anything V2, ZoeDepth, Depth Pro), metric3d.py, yolo.py, zipdepth.py,
    moge.py, unidepth.py, da3.py
  alerts/            alert logic (WP 3.1)
    corridor.py        walking corridor in 3D and alerts per image third
    naive.py           naive baseline without corridor
  eval/              evaluation framework (WP 5.1, 2.1)
    metrics.py         depth error (per depth range, binned by distance), alert metrics, latency
    alignment.py       raw, median and affine alignment (the last two are an oracle)
    timing.py          latency with warm-up and device synchronization, peak memory
    cache.py           prediction cache (results/cache/<model>/<frame>.npy, float16)
    runner.py          variants (rgb, intrinsics) -> (depth, alerts); runs models and fills the cache
    scoring.py         metrics from the cache, grouped by model, dataset, range and scene tag
    report.py          JSON and Markdown report
    pipeline.py        one run: data -> predictions -> metrics -> figures -> report; presets
  viz/               performance figures (matplotlib, optional): plots.py, style.py
scripts/             thin command line entry points
  run_eval.py          the evaluation (one command, see below)
  predict.py           fills the cache from a separate model environment
  check_recordings.py  checks names, manifest and frame alignment of the iPhone recordings
  setup_model_envs.sh  creates the separate environments for MoGe, UniDepth and DA3
  download_sanpo.py, calibrate.py
tests/               pytest with synthetic data and dummy models (no torch, no network, no weights)
```

The old module paths `blindspot.evaluation`, `blindspot.sanpo`, `blindspot.calibration` and
`blindspot.splits` still work: they re-export the functions from the new modules.

`data/`, `results/` and `envs/` in the repository root are git-ignored. Datasets, model weights and
results are never committed. Weights are cached in `~/.cache/blindspot/weights` (override with
`BLINDSPOT_WEIGHTS`), Hugging Face and torch hub use their own caches.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). The core (loaders, metrics, alert logic) only
needs numpy, OpenCV and Pillow; everything else is an extra:

| Extra | Contents |
|---|---|
| `models` | torch, torchvision, transformers, timm, ultralytics, mmengine (Metric3D) |
| `github` | ZipDepth from GitHub (pinned commit) |
| `viz` | matplotlib for the figures |
| `video` | PyAV, fallback decoder if OpenCV cannot read the HEVC videos |

**Linux PC with NVIDIA GPU** (final numbers):

```bash
uv sync --all-extras          # torch from PyPI ships with CUDA
uv run pytest                 # tests
```

**MacBook (Apple Silicon)**:

```bash
uv sync --all-extras          # torch from PyPI supports MPS
export PYTORCH_ENABLE_MPS_FALLBACK=1   # operators without an MPS kernel fall back to the CPU
uv run python scripts/run_eval.py --preset smoke
```

The device is chosen automatically (CUDA, then MPS, then CPU) or with `--device`. Nothing needs CUDA;
xFormers and TensorRT are not used. The Mac path is not tested; the CPU path is.

MoGe-2, UniDepthV2 and Depth Anything 3 need their own environments (conflicting pins, see
[docs/models.md](docs/models.md)): `bash scripts/setup_model_envs.sh`.

## Own recordings (WP 1.3)

Stray Scanner exports go to `data/iPhone/<recording>/` (`rgb.mp4`, `depth/`, `confidence/`,
`odometry.csv`, `camera_matrix.csv`). Every folder is named

```
<env>_<scene>_<light>_<obstacles>_<nn>        e.g. out_sidewalk_day_bollard-bike_01
```

| Part | Allowed values |
|---|---|
| `env` | `in`, `out`, `trans` |
| `scene` | `corridor`, `room`, `stairs`, `door`, `sidewalk`, `street`, `square`, `park`, `platform`, `underpass`, `garage`, `terrace` |
| `light` | `day`, `dusk`, `night`, `artificial`, `backlight` |
| `obstacles` | one or more of `pole`, `bollard`, `bike`, `bin`, `sign`, `person`, `car`, `glass`, `step` (also stairs), `branch`, `furniture`, `wall`, `plant`, `door`, `none`, joined with `-` (`none` alone) |
| `nn` | two digits, `01`, `02`, ... |

The parts become tags of every frame; reports group all metrics by them. Folders with other names are
skipped and listed in the report. An optional manifest `data/iPhone/recordings.csv` adds more:

```
recording,split,camera_height_m,weather,original_name,notes
out_sidewalk_day_bollard-bike_01,test,1.35,cloudy,1b8a88dae2,
```

`split` is `test` or `train`; with a manifest, the evaluation uses the `test` recordings only. Check the
folder with `uv run python scripts/check_recordings.py data/iPhone`.

Loader details: depth is millimeters at 256x192, only ARKit confidence 2 is kept, intrinsics come per frame
from `odometry.csv` (fallback `camera_matrix.csv`). Image, depth and intrinsics share one pixel grid,
3 x the LiDAR resolution (768x576): the image is downsampled, the depth upsampled with nearest neighbor.
The videos decode with one frame fewer than there are depth maps; the missing frame is the *first* one
(checked by correlating camera rotation with image motion), so video frame `k` belongs to depth frame
`k + 1`. LiDAR depth is unreliable on glass, at night and beyond about 5 m.

## Running the evaluation

```bash
# Smoke run: 3 SANPO frames + the first own recording (every 30th frame, at most 5), Tier-1 models
uv run python scripts/run_eval.py --preset smoke

# Full run: all downloaded SANPO test frames + all own test recordings (every 10th frame), all models
bash scripts/setup_model_envs.sh                                         # once
envs/moge/bin/python scripts/predict.py --models moge2_s --preset full
envs/unidepth/bin/python scripts/predict.py --models unidepth2_s --preset full
envs/da3/bin/python scripts/predict.py --models da3m_l,da3_s --preset full
uv run python scripts/run_eval.py --preset full --run-name m1_full

# Variants
uv run python scripts/run_eval.py --preset full --models mobile,m3d_s --datasets iphone --device cpu
uv run python scripts/run_eval.py --preset full --datasets sanpo --sanpo-split planning/splits/sanpo_real_test.txt \
    --annotations planning/annotations/sanpo_real_test.json
uv run python scripts/run_eval.py --preset smoke --models baseline        # constant 4 m lower bound
uv run python scripts/run_eval.py --list-models
```

Results of a run go to `results/<run-name>/` (default `<preset>_<date>_<time>`):

| File | Contents |
|---|---|
| `report.md` | tables (performance, accuracy per alignment and depth range, scene tags, alerts) with the figures linked |
| `report.json` | configuration, environment, model status and performance, all aggregated and per-frame metrics |
| `figures/` | per dataset, each as PNG (slides) and PDF (LaTeX): accuracy vs. speed, raw vs. scaled, accuracy by distance range, error over distance (models from `highlight` in the config), day/night and indoor/outdoor (own recordings) |

Predictions are cached in `results/cache/<model>/` (float16), so a second run only recomputes metrics,
figures and report (seconds). `--recompute` ignores the cache. A model that fails to load or run is listed
with its reason and the run continues.

What is measured: AbsRel, RMSE and δ1 per depth range (0-2, 2-5, 5-10 m, all) with three alignments
(`raw` for metric models; `median` and `affine` use the ground truth and are an oracle upper bound),
AbsRel binned over distance, parameters, load time, latency (mean, p95, FPS, after warm-up, device
synchronized) and peak memory, and alert metrics (recall, false alarm rate, direction, distance error) of
"metric model + corridor" and "+ naive baseline" once obstacle annotations exist.
