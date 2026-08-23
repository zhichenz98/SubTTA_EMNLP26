#!/bin/bash
# Sequentially run all paper methods and collect results.
# Usage: bash sh/run_all_experiments.sh [GPU_ID]
set -uo pipefail

GPU_ID="${1:-4}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%y%m%d_%H%M%S)"
LOG_ROOT="${ROOT}/classification/output/run_all_${STAMP}"
mkdir -p "${LOG_ROOT}"
SUMMARY="${LOG_ROOT}/summary.tsv"
ERRORS="${LOG_ROOT}/errors.log"

echo -e "method\tdataset\tstatus\tmean_error\tmean_error_at_5\tlogfile" > "${SUMMARY}"
echo "Logs: ${LOG_ROOT}"
echo "GPU: ${GPU_ID}"

run_job() {
  local name="$1"
  shift
  local logfile="${LOG_ROOT}/${name}.log"
  echo "============================================================"
  echo "[$(date '+%F %T')] START ${name}"
  echo "============================================================"
  (
    cd "${ROOT}/classification"
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    python test_time.py "$@" MODEL.ARCH ViT-B-16 MODEL.WEIGHTS openai MODEL.USE_CLIP True
  ) >"${logfile}" 2>&1
  local rc=$?

  # Parse mean error lines from log
  local mean mean5 status
  mean="$(grep -oE 'mean error: [0-9.]+%' "${logfile}" | tail -1 | grep -oE '[0-9.]+%' || true)"
  mean5="$(grep -oE 'mean error at 5: [0-9.]+%' "${logfile}" | tail -1 | grep -oE '[0-9.]+%' || true)"
  if [[ ${rc} -eq 0 && -n "${mean}" ]]; then
    status="OK"
  else
    status="FAIL"
    {
      echo "===== ${name} rc=${rc} ====="
      tail -80 "${logfile}"
      echo
    } >> "${ERRORS}"
  fi

  # Infer dataset from cfg path in args
  local dataset="unknown"
  for a in "$@"; do
    if [[ "${a}" == *cifar10_c* ]]; then dataset="cifar10_c"; fi
    if [[ "${a}" == *cifar100_c* ]]; then dataset="cifar100_c"; fi
    if [[ "${a}" == *imagenet_c* ]]; then dataset="imagenet_c"; fi
  done

  echo -e "${name}\t${dataset}\t${status}\t${mean:-NA}\t${mean5:-NA}\t${logfile}" >> "${SUMMARY}"
  echo "[$(date '+%F %T')] END ${name} status=${status} mean=${mean:-NA} mean@5=${mean5:-NA}"
  return ${rc}
}

# Do not abort the whole suite on a single failure
set +e

# ---- Baselines ----
for ds in cifar10_c cifar100_c imagenet_c; do
  run_job "source__${ds}" --cfg "cfgs/${ds}/source.yaml"
done

for ds in cifar10_c cifar100_c imagenet_c; do
  run_job "tent__${ds}" --cfg "cfgs/${ds}/tent.yaml"
done

for ds in cifar10_c cifar100_c imagenet_c; do
  run_job "mint__${ds}" --cfg "cfgs/${ds}/mint.yaml"
done

for ds in cifar10_c cifar100_c imagenet_c; do
  run_job "batclip__${ds}" --cfg "cfgs/${ds}/batclip.yaml"
done

# ---- SubTTA x {source,tent,mint,batclip} ----
for loss in source tent mint batclip; do
  for ds in cifar10_c cifar100_c imagenet_c; do
    run_job "subtta_${loss}__${ds}" --cfg "cfgs/${ds}/subtta.yaml" SUBTTA.LOSS_TYPE "${loss}"
  done
done

echo
echo "======= DONE ======="
echo "Summary: ${SUMMARY}"
echo "Errors:  ${ERRORS}"
column -t -s $'\t' "${SUMMARY}" || cat "${SUMMARY}"
