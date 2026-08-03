#!/usr/bin/env bash
set -e
set -o pipefail

PROJ="${PROJ:-/workspace/mlir_project}"
BUILD="${BUILD:-${PROJ}/build-ninja}"
MLIR_BUILD_DIR="${MLIR_BUILD_DIR:-/build/build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_CHIP="${CUDA_CHIP:-sm_89}"
CUDA_PTX_FEATURE="${CUDA_PTX_FEATURE:-+ptx80}"
GPU_MAPPING_POLICY="${GPU_MAPPING_POLICY:-innermost-first}"
GPU_LOWERING="${GPU_LOWERING:-block-thread}"
BLOCK_M="${BLOCK_M:-8}"
BLOCK_N="${BLOCK_N:-32}"
TILE_M="${TILE_M:-16}"
TILE_N="${TILE_N:-16}"
TILE_K="${TILE_K:-256}"

export PATH="${MLIR_BUILD_DIR}/bin:${BUILD}/tools:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MLIR_BUILD_DIR}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

cd "${PROJ}/src/sample/gpu"

MODEL_MLIR="${PROJ}/src/benchmarks/large_gemm.mlir" \
GPU_LOWERING="${GPU_LOWERING}" BLOCK_M="${BLOCK_M}" BLOCK_N="${BLOCK_N}" \
TILE_M="${TILE_M}" TILE_N="${TILE_N}" TILE_K="${TILE_K}" \
GPU_MAPPING_POLICY="${GPU_MAPPING_POLICY}" \
CUDA_CHIP="${CUDA_CHIP}" CUDA_PTX_FEATURE="${CUDA_PTX_FEATURE}" \
bash run_mlir_pipeline.sh

SAMPLE_CALL="${PROJ}/src/benchmarks/large_gemm_gpu_bench.cpp" bash compile.sh

if [[ "${CHECK_LAUNCHES:-0}" == "1" ]]; then
  MLIR_CUDA_DEBUG=1 ./a.out
else
  ./a.out
fi
