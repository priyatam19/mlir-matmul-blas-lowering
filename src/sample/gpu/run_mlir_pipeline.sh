#!/usr/bin/env bash
set -e
set -o pipefail

CUDA_CHIP="${CUDA_CHIP:-sm_75}"
CUDA_PTX_FEATURE="${CUDA_PTX_FEATURE:-+ptx75}"
MODEL_MLIR="${MODEL_MLIR:-sample_model_linalg.mlir}"
TILE_M="${TILE_M:-2}"
TILE_N="${TILE_N:-4}"
TILE_K="${TILE_K:-4}"
GPU_MAPPING_POLICY="${GPU_MAPPING_POLICY:-outermost-first}"

../../../build-ninja/tools/tutorial-opt \
  --tile-matmul-for-cache="tile-m=${TILE_M} tile-n=${TILE_N} tile-k=${TILE_K}" \
  "${MODEL_MLIR}" \
| mlir-opt \
  --convert-tensor-to-linalg \
  --linalg-generalize-named-ops \
  --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  --buffer-deallocation-pipeline \
  --convert-bufferization-to-memref \
  --llvm-request-c-wrappers \
  --convert-linalg-to-parallel-loops \
  --gpu-map-parallel-loops="mapping-policy=${GPU_MAPPING_POLICY}" \
  --convert-parallel-loops-to-gpu \
  --canonicalize \
  --cse \
| mlir-opt \
  --gpu-kernel-outlining \
  --lower-affine \
  --gpu-decompose-memrefs \
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
  -o sample_nvptx_isa.mlir

mlir-translate -mlir-to-llvmir sample_nvptx_isa.mlir -o sample.ll

llc -filetype=obj -O3 sample.ll
