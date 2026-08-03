#!/usr/bin/env bash
set -e
set -o pipefail

### Pipeline to demonstrate the custom tensor-level matmul tiling pass. ###

MODEL_MLIR="${MODEL_MLIR:-$PWD/sample_model_linalg.mlir}"
SAMPLE_CALL="${SAMPLE_CALL:-sample_call.cpp}"
OUT="${OUT:-tiled.out}"

../../build-ninja/tools/tutorial-opt \
  --tile-matmul-for-cache="tile-m=2 tile-n=4 tile-k=4" \
  --linalg-to-bufferization \
  "${MODEL_MLIR}" > $PWD/sample_model_tiled_buf_linalg.mlir

../../build-ninja/tools/tutorial-opt \
  --llvm-request-c-wrappers \
  --bufferization-to-llvm \
  $PWD/sample_model_tiled_buf_linalg.mlir > $PWD/sample_model_tiled_llvm.mlir

mlir-translate -mlir-to-llvmir \
  $PWD/sample_model_tiled_llvm.mlir > $PWD/sample_model_tiled_llvm_ir.ll

llc --filetype=obj $PWD/sample_model_tiled_llvm_ir.ll

OPENBLAS_FLAGS="-lopenblas"
if [[ -d "../../openblas/lib" ]]; then
  OPENBLAS_FLAGS="-L../../openblas/lib -lopenblas"
fi

g++ -O3 -c "${SAMPLE_CALL}" -o sample_call.o && \
  g++ -O3 -no-pie sample_call.o sample_model_tiled_llvm_ir.o -o "${OUT}" \
    ${OPENBLAS_FLAGS} -lm
