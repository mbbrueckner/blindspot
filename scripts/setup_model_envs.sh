#!/usr/bin/env bash
# Creates the separate environments for models whose dependencies conflict with the main environment
# (docs/models.md). Each env gets torch, the blindspot core package and the model package installed
# with --no-deps plus only the modules its inference code imports, all pinned to tested commits.
#
# Usage: bash scripts/setup_model_envs.sh [moge] [unidepth] [da3]   (no argument: all three)
# Then fill the prediction cache, e.g.: envs/moge/bin/python scripts/predict.py --models moge2_s --preset full
set -euo pipefail
cd "$(dirname "$0")/.."

PY=3.12
MOGE="moge @ git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a"
UTILS3D="utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37"
UNIDEPTH="unidepth @ git+https://github.com/lpiccinelli-eth/UniDepth.git@8d8cfe4c7ee15297099983607febf0d4f32eb3d6"
DA3="depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"

make_env() {  # name, extra packages...
  local name=$1; shift
  uv venv -q --allow-existing "envs/$name" --python "$PY"
  uv pip install -q --python "envs/$name/bin/python" torch torchvision huggingface-hub -e . "$@"
}

setup_moge() {
  make_env moge scipy
  uv pip install -q --python envs/moge/bin/python --no-deps "$MOGE" "$UTILS3D"
  envs/moge/bin/python -c "from moge.model.v2 import MoGeModel" && echo "moge: ok"
}

setup_unidepth() {
  # UniDepth pins torch and requires xformers/triton; both are optional at inference (guarded imports).
  make_env unidepth einops timm scipy matplotlib wandb
  uv pip install -q --python envs/unidepth/bin/python --no-deps "$UNIDEPTH"
  envs/unidepth/bin/python -c "from unidepth.models import UniDepthV2" && echo "unidepth: ok"
}

setup_da3() {
  # DA3 declares Python <= 3.13, numpy < 2 and xformers; inference works with numpy 2 and without xformers.
  # moviepy must be 1.x (moviepy.editor). gsplat (3D Gaussian rendering) is not needed for depth.
  make_env da3 einops safetensors omegaconf "moviepy==1.0.3" addict plyfile matplotlib pycolmap trimesh evo
  uv pip install -q --python envs/da3/bin/python --no-deps "$DA3"
  envs/da3/bin/python -c "from depth_anything_3.api import DepthAnything3" 2>/dev/null && echo "da3: ok"
}

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(moge unidepth da3)
for t in "${targets[@]}"; do "setup_$t"; done
