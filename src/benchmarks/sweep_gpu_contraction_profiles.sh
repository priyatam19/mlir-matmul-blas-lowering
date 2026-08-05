#!/usr/bin/env bash
set -euo pipefail

PROJ="${PROJ:-/workspace/mlir_project}"
RESULTS_FILE="${RESULTS_FILE:-${PROJ}/runpod_results/contraction_profile_sweep.csv}"
SHAPES="${SHAPES:-gemm_512 gemm_1024}"
WARMUPS="${WARMUPS:-10}"
RUNS="${RUNS:-50}"
mkdir -p "$(dirname "${RESULTS_FILE}")"
: >"${RESULTS_FILE}"

profiles=(
  "64 64 16 128 1"
  "64 128 16 256 1"
  "128 64 16 256 1"
  "128 128 16 256 1"
  "128 128 16 256 2"
  "64 128 32 256 2"
  "128 64 32 256 2"
)

for mode in shared-fp32 tensorcore-tf32; do
  for profile in "${profiles[@]}"; do
    read -r block_m block_n block_k threads stages <<<"${profile}"
    echo "mode=${mode} profile=${block_m}x${block_n}x${block_k} threads=${threads} stages=${stages}" >&2
    PROJ="${PROJ}" GPU_LOWERING="${mode}" SHAPES="${SHAPES}" \
      CONTRACTION_BLOCK_M="${block_m}" CONTRACTION_BLOCK_N="${block_n}" \
      CONTRACTION_BLOCK_K="${block_k}" CONTRACTION_THREADS="${threads}" \
      CONTRACTION_STAGES="${stages}" WARMUPS="${WARMUPS}" RUNS="${RUNS}" \
      RESULTS_FILE="${RESULTS_FILE}" \
      bash "${PROJ}/src/benchmarks/run_gpu_gemm_benchmarks.sh"
  done
done
