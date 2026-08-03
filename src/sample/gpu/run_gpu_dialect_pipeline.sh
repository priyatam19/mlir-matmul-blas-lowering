#!/usr/bin/env bash
set -e
set -o pipefail

../../../build-ninja/tools/tutorial-opt \
  --tile-matmul-for-cache="tile-m=2 tile-n=4 tile-k=4" \
  sample_model_linalg.mlir \
| mlir-opt \
  --convert-tensor-to-linalg \
  --linalg-generalize-named-ops \
  --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  --buffer-deallocation-pipeline \
  --convert-bufferization-to-memref \
  --llvm-request-c-wrappers \
  --convert-linalg-to-parallel-loops \
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --canonicalize \
  --cse \
| mlir-opt \
  --gpu-kernel-outlining \
  --lower-affine \
  --gpu-decompose-memrefs \
  --canonicalize \
  --cse \
  -o sample_gpu_dialect.mlir
