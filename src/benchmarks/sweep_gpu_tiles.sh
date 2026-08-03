#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_FILE="${RESULTS_FILE:-${SCRIPT_DIR}/gpu_tile_sweep.csv}"

: >"${RESULTS_FILE}"
for tile in 8x32 16x16 16x32 32x8; do
  block_m="${tile%x*}"
  block_n="${tile#*x}"
  echo "Sweeping block ${block_m}x${block_n}" >&2
  SHAPES="gemm_512 gemm_1024" GPU_LOWERING=block-thread \
    BLOCK_M="${block_m}" BLOCK_N="${block_n}" \
    RESULTS_FILE="${RESULTS_FILE}" \
    bash "${SCRIPT_DIR}/run_gpu_gemm_benchmarks.sh"
done

echo "Wrote ${RESULTS_FILE}."
