#!/bin/bash
# BATCLIP baseline
# Usage: bash sh/batclip.sh [GPU_ID]
set -euo pipefail
GPU_ID="${1:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/classification"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

for cfg in cfgs/cifar10_c/batclip.yaml cfgs/cifar100_c/batclip.yaml cfgs/imagenet_c/batclip.yaml; do
  echo "[batclip] Running ${cfg} on GPU ${GPU_ID}"
  python test_time.py --cfg "${cfg}" \
    MODEL.ARCH ViT-B-16 MODEL.WEIGHTS openai MODEL.USE_CLIP True
done
