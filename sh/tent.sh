#!/bin/bash
# TENT baseline
# Usage: bash sh/tent.sh [GPU_ID]
set -euo pipefail
GPU_ID="${1:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/classification"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

for cfg in cfgs/cifar10_c/tent.yaml cfgs/cifar100_c/tent.yaml cfgs/imagenet_c/tent.yaml; do
  echo "[tent] Running ${cfg} on GPU ${GPU_ID}"
  python test_time.py --cfg "${cfg}" \
    MODEL.ARCH ViT-B-16 MODEL.WEIGHTS openai MODEL.USE_CLIP True
done
