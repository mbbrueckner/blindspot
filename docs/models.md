# Depth models for WP 2.1

Overview of all models in `blindspot/models/`, how they are integrated and what we found while
integrating them. As of 23.11.2026. Short names and groups: `uv run python scripts/run_eval.py --list-models`.

## Overview

| Name | Model | Type | uses K | Params | Environment | Source (pinned) |
|---|---|---|---|---|---|---|
| `midas_s` | MiDaS v2.1 Small | relative (disparity) | no | 21.3 M | main | torch hub `intel-isl/MiDaS:v3_1` |
| `da2_s` | Depth Anything V2 Small | relative (disparity) | no | 24.8 M | main | HF `depth-anything/Depth-Anything-V2-Small-hf` |
| `da2m_in_s` | Depth Anything V2 Metric Indoor Small | metric | no | 24.8 M | main | HF `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` |
| `da2m_out_s` | Depth Anything V2 Metric Outdoor Small | metric | no | 24.8 M | main | HF `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` |
| `m3d_s` | Metric3D v2 ViT-S | metric | yes | 37.5 M | main | torch hub `yvanyin/metric3d@eb5b6fa` |
| `yolo26n` | YOLO26n-Depth | metric | no | 6.4 M | main | Ultralytics `yolo26n-depth.pt` |
| `zipdepth` | ZipDepth Base | relative (disparity) | no | 6.1 M | main (extra `github`) | GitHub `fabiotosi92/ZipDepth@91f3fd2` |
| `moge2_s` | MoGe-2 ViT-S (normal) | metric | yes (FOV) | 35.1 M | `envs/moge` | HF `Ruicheng/moge-2-vits-normal`, code `microsoft/MoGe@74fbce0` |
| `unidepth2_s` | UniDepthV2 ViT-S | metric | yes | 34.2 M | `envs/unidepth` | HF `lpiccinelli/unidepth-v2-vits14`, code `lpiccinelli-eth/UniDepth@8d8cfe4` |
| `depthpro` | Apple Depth Pro | metric | yes | 952 M | main | HF `apple/DepthPro-hf` |
| `da3m_l` | Depth Anything 3 Metric Large | metric | yes | 334 M | `envs/da3` | HF `depth-anything/DA3METRIC-LARGE`, code `ByteDance-Seed/Depth-Anything-3@3d835ec` |
| `da3_s` | Depth Anything 3 Small | relative (depth) | no | 34.3 M | `envs/da3` | HF `depth-anything/DA3-SMALL` |
| `zoedepth` | ZoeDepth NYU+KITTI | metric | no | 345 M | main | HF `Intel/zoedepth-nyu-kitti` |
| `m3d_l` | Metric3D v2 ViT-L | metric | yes | 412 M | main | torch hub `yvanyin/metric3d@eb5b6fa` |
| `constant` | Constant 6 m (lower bound) | metric | no | – | main | none |

Groups: `tier1` (the first seven), `tier2` (the next seven), `baseline` (`constant`, the lower bound every
model has to beat), `mobile` (below 30 M parameters), `camera` (use our focal length), `reference` (large
models as an accuracy reference), `metric`, `relative`, `main` (run in the main environment), `all`.

The models fall into three groups by where the meters come from:

1. **Metric without camera information** (DA2 Metric, YOLO26n, ZoeDepth): the scale is learned from the
   training images, so it silently assumes our camera is like the training camera.
2. **Metric with focal length** (Metric3D, UniDepth, MoGe, DA3 Metric, Depth Pro): the network predicts depth
   for a canonical camera and converts it with the focal length we pass in.
3. **Relative** (MiDaS, DA2 Small, ZipDepth, DA3 Small): only the ordering; scale (and for disparity models
   also a shift) must come from outside, e.g. from the ground plane and the camera height (WP 2.2).

## Common interface

Every model gets `predict(rgb, intrinsics)` and returns depth at the resolution of the image:

- **Metric models** return meters. Models that use the focal length get our intrinsics, converted to their
  input resolution.
- **Relative models** return disparity with unknown scale *and* shift. The wrapper divides it by its median
  and returns `1 / disparity` as pseudo-depth. For scoring it is aligned per frame (affine in disparity, see
  the report). A single scale factor (median) is not enough for relative models because it cannot remove
  the shift: on an iPhone frame MiDaS has AbsRel 1.07 with median scaling and 0.13 with affine alignment.
- No model sees the reference depth.

## Check on real images

Each model ran on a real SANPO frame (chest camera, 720x1280, reference median 4.69 m): shape `(720, 1280)`,
float32, no NaN, plausible values. δ1 for the three alignments (`raw` only meaningful for metric models).
Latency for a single frame after one warm-up run, RTX 5070 Ti or CPU (the proper comparison is in the run
report).

| Model | Median prediction [m] | δ1 raw | δ1 median | δ1 affine | CUDA [ms] | CPU [ms] |
|---|---|---|---|---|---|---|
| `midas_s` | (relative) | – | 0.921 | 0.978 | 12 | 17 |
| `da2_s` | (relative) | – | 0.959 | 0.997 | 32 | 365 |
| `da2m_in_s` | 3.47 | 0.121 | 0.997 | 0.997 | 27 | 396 |
| `da2m_out_s` | 11.40 | 0.000 | 0.919 | 0.956 | 28 | 389 |
| `m3d_s` | 5.34 | **0.989** | 0.997 | 0.999 | 74 | 1188 |
| `yolo26n` | 3.55 | 0.183 | 0.967 | 0.992 | 8 | 38 |
| `zipdepth` | (relative) | – | 0.743 | 0.998 | 7 | 27 |
| `moge2_s` | 6.55 | 0.007 | 0.997 | 0.992 | 64 | – |
| `unidepth2_s` | 2.70 | 0.001 | 0.996 | 0.997 | 26 | – |
| `depthpro` | 0.92 | 0.000 | 0.925 | 0.949 | 319 | – |
| `da3m_l` | 5.33 | **0.985** | 0.996 | 0.997 | 26 | – |
| `da3_s` | (relative) | – | 0.997 | 0.997 | 45 | – |
| `zoedepth` | 2.59 | 0.000 | 0.942 | 0.993 | 121 | – |
| `m3d_l` | 4.80 | **0.994** | 0.994 | 0.887 | 369 | – |

All Tier-1 models run on CUDA and on the CPU with the same results (δ1 equal to three decimals). Tier 2 is
only checked on CUDA. On this frame only the models that use our focal length and are trained on many cameras
(Metric3D, DA3 Metric) get the metric scale right. This is a single frame and only a plausibility check; the
full run over the test set (`results/m1_full/`) gives the reliable numbers.

## Notes per model

- **MiDaS Small:** `MiDaS_small` loads its EfficientNet backbone with a second `torch.hub.load`
  (`rwightman/gen-efficientnet-pytorch`) without `trust_repo`. Torch then asks on stdin whether to trust the
  repo, and a run without a terminal hangs (this happened to us). The wrapper trusts the repo up front.
- **Depth Anything V2 Metric:** Indoor is fine-tuned on Hypersim (up to 20 m), Outdoor on Virtual KITTI 2
  (up to 80 m, camera on a car). Neither uses the focal length. Outdoor predicts much too far on our images
  (SANPO frame: median 11.4 m instead of 4.7 m).
- **Metric3D v2:** The hub code assumes CUDA and mmcv. Two fixes in `blindspot/models/metric3d.py`, without
  changing the hub code: (1) `mono/utils/comm.py` imports `mmcv.utils.collect_env` (only used for logging); if
  mmcv is missing, a stub is registered and the config loads via `mmengine` (extra `models`).
  (2) `RAFTDepthNormalDPT5.get_bins` creates tensors with `device="cuda"` and is replaced by a version that
  uses the device of the weights. Metric3D thus runs on CUDA, CPU and (untested) MPS. xFormers is optional.
  Preprocessing as in `hubconf.py`: fit into 616x1064, pad, scale the focal length along, depth = output ·
  fx / 1000.
- **YOLO26n-Depth:** part of Ultralytics (`YOLO("yolo26n-depth.pt")`, result in `results[0].depth.data`,
  meters). The weights are trained at `imgsz=768`, so we predict at that size. The measured latency includes
  the Ultralytics predictor (pre- and post-processing). Source: https://docs.ultralytics.com/tasks/depth
- **ZipDepth:** ECCV 2026, University of Bologna, https://github.com/fabiotosi92/ZipDepth (paper arXiv
  2607.08771). The package contains only code with no dependency besides torch and is installed via the
  `github` extra from the pinned commit; the wrapper downloads the weights (`zipdepth_base.pth`, 27 MB) from
  the repo on first use. Preprocessing as in `zipdepth/inference/predictor.py`: shorter side 384 px, sides
  multiples of 32, RGB in [0, 1], normalization inside the network. Output is affine-invariant inverse depth.
  The weights are distilled from Depth Anything V2 Large (mind the license of the weights).
- **Depth Pro:** via `transformers` (`apple/DepthPro-hf`). The network returns a canonical inverse depth `c`;
  meters are `f_px / (width · c)`. The official post-processing takes `f_px` from the field of view the model
  estimates; we use our fx. On the SANPO test frame Depth Pro is 5x too near; this is not caused by the
  conversion, since the official post-processing gives the same result (0.86 m) and the estimated field of view
  is almost right (63.5° instead of 59.9°). On an indoor iPhone frame the scale is right (median 3.08 m instead
  of 2.80 m). Presumably squashing every image to 1536x1536 distorts SANPO's 16:9 frames more than the
  iPhone's 4:3. In front of walls it predicts single pixels hundreds of meters away, which is why predictions
  are clipped to 10 m before scoring.
- **ZoeDepth:** via `transformers`; the official post-processing removes the padding. It already routes
  between an indoor and an outdoor head internally (see below).
- **MoGe-2:** gets our horizontal field of view (`fov_x`), otherwise it estimates it. fp16 only on CUDA.
- **UniDepthV2:** gets the pinhole matrix at image resolution; the model converts it to its input size.
- **Depth Anything 3:** metric depth = focal length · output / 300, with the focal length (mean of fx and fy)
  at the network resolution (`process_res=504`). Source: model card
  https://huggingface.co/depth-anything/DA3METRIC-LARGE. The check above (δ1 raw 0.985) confirms the formula.
  `DA3-SMALL` returns relative *depth* (not disparity).

## Indoor/outdoor routing

Idea: a simple classifier decides whether the camera is inside or outside and routes to a dedicated model.
The full run (197 indoor, 247 outdoor, 30 transition frames) speaks against it for our models:

- Only Depth Anything V2 Metric has separate indoor/outdoor variants, and there routing would hurt: on our
  outdoor recordings the indoor variant is better (raw δ1 0.46) than the outdoor variant (0.09); on SANPO
  streets both are poor (0.15 and 0.18). The outdoor variant is trained for car cameras up to 80 m.
- ZoeDepth already implements exactly this routing internally and is still poor outdoors (raw δ1 0.35).
- The models that use the focal length do not need it: DA3 Metric Large is equally good inside and outside
  (0.86 / 0.85), and after median scaling Metric3D, UniDepth and MoGe are equal inside and outside (0.89–0.91).
- For relative models it does not matter, the scale comes from ground scaling anyway.
- Downsides: two models in memory on the phone, flickering at doors (transition frames were already the
  hardest), wrong model on misclassification.

A cheap environment or day/night signal can still help: some models have a constant scale bias per
environment (UniDepth needs a factor of 0.92 inside and 0.78 outside on median), and night hurts more than
indoor/outdoor (median-scaled δ1 of Metric3D S drops from 0.93 by day to 0.83 at night).

## Separate environments (MoGe, UniDepth, DA3)

These three do not fit into the main environment (Python 3.14, numpy 2):

- **DA3** requires Python ≤ 3.13, `numpy<2` and xformers.
- **UniDepth** pins torch and requires xformers and triton (Linux only).
- **MoGe** pulls in gradio, trimesh and `flex-gemm` (a CUDA extension that has to be compiled).

Inference needs little of this: MoGe-2 uses `flex-gemm` only in v3, the xformers imports of UniDepth and DA3
are guarded with `try`, and DA3 also runs with numpy 2. So each model gets its own environment with Python
3.12, torch, the `blindspot` core package and the model package installed with `--no-deps`, plus only the
modules its inference code imports:

```bash
bash scripts/setup_model_envs.sh            # all three, or: ... moge | unidepth | da3
envs/moge/bin/python scripts/predict.py --models moge2_s --preset full
envs/unidepth/bin/python scripts/predict.py --models unidepth2_s --preset full
envs/da3/bin/python scripts/predict.py --models da3m_l,da3_s --preset full
uv run python scripts/run_eval.py --preset full      # finds the predictions in the cache
```

`predict.py` loads the same frames as `run_eval.py` with the same preset and writes predictions and timings
to `results/cache/<model>/`. If the cache is missing, `run_eval.py` reports the model as failed (with an
install hint) and continues with the others. On the Mac, triton is missing for UniDepth; the imports are
guarded, but this is untested.

## Mac (Apple Silicon)

All wrappers choose the device through `blindspot/device.py` (CUDA, then MPS, then CPU) and do not require
CUDA. On the Mac, set `export PYTORCH_ENABLE_MPS_FALLBACK=1` first, so operators without an MPS kernel fall
back to the CPU. Not tested on the Mac; the CPU runs serve as a stand-in. Expected pitfalls: Depth Pro
(952 M parameters, fp32) needs about 4 GB, Metric3D-L and DA3-L more than 1.5 GB each. MPS peak memory is
only sampled after each frame (there is no counter as on CUDA).
