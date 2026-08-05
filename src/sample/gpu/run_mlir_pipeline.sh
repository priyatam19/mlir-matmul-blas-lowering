#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BUILD="${BUILD:-${PROJECT_ROOT}/build-ninja}"
TUTORIAL_OPT="${BUILD}/tools/tutorial-opt"

CUDA_CHIP="${CUDA_CHIP:-sm_75}"
CUDA_PTX_FEATURE="${CUDA_PTX_FEATURE:-+ptx75}"
MODEL_MLIR="${MODEL_MLIR:-${SCRIPT_DIR}/sample_model_linalg.mlir}"
GPU_LOWERING="${GPU_LOWERING:-block-thread}"
GPU_MAPPING_POLICY="${GPU_MAPPING_POLICY:-innermost-first}"
BLOCK_M="${BLOCK_M:-8}"
BLOCK_N="${BLOCK_N:-32}"
TILE_M="${TILE_M:-16}"
TILE_N="${TILE_N:-16}"
TILE_K="${TILE_K:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}}"
mkdir -p "${OUTPUT_DIR}"

BUFFERIZED_MLIR="${OUTPUT_DIR}/sample_gpu_bufferized.mlir"
GPU_DIALECT_MLIR="${OUTPUT_DIR}/sample_gpu_dialect.mlir"
NVPTX_MLIR="${OUTPUT_DIR}/sample_nvptx_isa.mlir"

bufferize_common=(
  --convert-tensor-to-linalg
  --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
  --buffer-deallocation-pipeline
  --convert-bufferization-to-memref
  --llvm-request-c-wrappers
)

case "${GPU_LOWERING}" in
  legacy)
    "${TUTORIAL_OPT}" \
      --tile-matmul-for-cache="tile-m=${TILE_M} tile-n=${TILE_N} tile-k=${TILE_K}" \
      "${MODEL_MLIR}" \
    | mlir-opt \
        --convert-tensor-to-linalg \
        --linalg-generalize-named-ops \
        --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
        --buffer-deallocation-pipeline \
        --convert-bufferization-to-memref \
        --llvm-request-c-wrappers \
        -o "${BUFFERIZED_MLIR}"
    ;;
  untiled)
    mlir-opt "${MODEL_MLIR}" \
      --convert-tensor-to-linalg \
      --linalg-generalize-named-ops \
      --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
      --buffer-deallocation-pipeline \
      --convert-bufferization-to-memref \
      --llvm-request-c-wrappers \
      -o "${BUFFERIZED_MLIR}"
    ;;
  block-thread)
    mlir-opt "${MODEL_MLIR}" "${bufferize_common[@]}" \
    | "${TUTORIAL_OPT}" \
        --tile-matmul-for-gpu="block-m=${BLOCK_M} block-n=${BLOCK_N}" \
        --tile-batch-matmul-for-gpu="block-m=${BLOCK_M} block-n=${BLOCK_N}" \
        -o "${BUFFERIZED_MLIR}"
    ;;
  vendor)
    mlir-opt "${MODEL_MLIR}" "${bufferize_common[@]}" \
    | "${TUTORIAL_OPT}" \
        --convert-batch-matmul-to-cublas \
        -o "${BUFFERIZED_MLIR}"
    ;;
  *)
    echo "Unknown GPU_LOWERING=${GPU_LOWERING}; expected legacy, untiled, block-thread, or vendor." >&2
    exit 2
    ;;
esac

mlir-opt "${BUFFERIZED_MLIR}" \
  --convert-linalg-to-parallel-loops \
  --gpu-map-parallel-loops="mapping-policy=${GPU_MAPPING_POLICY}" \
  --convert-parallel-loops-to-gpu \
  --canonicalize \
  --cse \
  --gpu-kernel-outlining \
  --lower-affine \
  --gpu-decompose-memrefs \
  --canonicalize \
  --cse \
  -o "${GPU_DIALECT_MLIR}"

if [[ "${GPU_DIALECT_ONLY:-0}" == "1" ]]; then
  echo "Generated ${GPU_DIALECT_MLIR} with GPU_LOWERING=${GPU_LOWERING}."
  exit 0
fi

mlir-opt "${GPU_DIALECT_MLIR}" \
  --expand-strided-metadata \
  --normalize-memrefs \
  --convert-index-to-llvm \
  --arith-expand \
  --memref-expand \
  --gpu-lower-to-nvvm-pipeline="cubin-chip=${CUDA_CHIP} cubin-features=${CUDA_PTX_FEATURE} cubin-format=isa opt-level=3" \
  --convert-nvvm-to-llvm \
  --reconcile-unrealized-casts \
  --gpu-to-llvm='use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true' \
  --gpu-module-to-binary \
  -o "${NVPTX_MLIR}"

mlir-translate -mlir-to-llvmir "${NVPTX_MLIR}" -o "${OUTPUT_DIR}/sample.ll"
llc -filetype=obj -O3 "${OUTPUT_DIR}/sample.ll" -o "${OUTPUT_DIR}/sample.o"

echo "Generated ${OUTPUT_DIR}/sample.o with GPU_LOWERING=${GPU_LOWERING}."
