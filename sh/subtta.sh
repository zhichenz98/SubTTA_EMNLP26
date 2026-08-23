#!/bin/bash
# SubTTA with separation losses: source / tent / mint / batclip
# Usage: bash sh/subtta.sh [GPU_ID] [LOSS_TYPE]
#   LOSS_TYPE default: mint
#   Example: bash sh/subtta.sh 0 tent
set -euo pipefail
GPU_ID="${1:-4}"
LOSS_TYPE="${2:-mint}"  # source | tent | mint | batclip
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/classification"

case "${LOSS_TYPE}" in
  source|tent|mint|batclip) ;;
  *)
    echo "Unsupported LOSS_TYPE=${LOSS_TYPE}. Choose from: source tent mint batclip" >&2
    exit 1
    ;;
esac

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

for cfg in cfgs/cifar10_c/subtta.yaml cfgs/cifar100_c/subtta.yaml cfgs/imagenet_c/subtta.yaml; do
  echo "[subtta] Running ${cfg} LOSS_TYPE=${LOSS_TYPE} on GPU ${GPU_ID}"
  python test_time.py --cfg "${cfg}" \
    SUBTTA.LOSS_TYPE "${LOSS_TYPE}" \
    MODEL.ARCH ViT-B-16 MODEL.WEIGHTS openai MODEL.USE_CLIP True
done
