#!/usr/bin/env bash
set -e
set -o pipefail

MLIR_BUILD_DIR="${MLIR_BUILD_DIR:-/build/build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
SAMPLE_CALL="${SAMPLE_CALL:-sample_call.cpp}"

if [[ ! -f sample.o ]]; then
  echo "sample.o is missing. Run: bash run_mlir_pipeline.sh" >&2
  exit 1
fi

if [[ ! -e "${CUDA_HOME}/lib64/libcuda.so" &&
      ! -e "${CUDA_HOME}/lib64/stubs/libcuda.so" ]]; then
  echo "CUDA driver library not found under CUDA_HOME=${CUDA_HOME}." >&2
  echo "Set CUDA_HOME to a CUDA toolkit install before linking the GPU runner." >&2
  exit 1
fi

if [[ ! -e "${CUDA_HOME}/lib64/libcudart.so" ]]; then
  echo "CUDA runtime library not found under CUDA_HOME=${CUDA_HOME}." >&2
  exit 1
fi

if [[ ! -e "${MLIR_BUILD_DIR}/lib/libmlir_cuda_runtime.a" &&
      ! -e "${MLIR_BUILD_DIR}/lib/libmlir_cuda_runtime.so" ]]; then
  echo "MLIR CUDA runtime library not found under MLIR_BUILD_DIR=${MLIR_BUILD_DIR}." >&2
  echo "Rebuild MLIR with the CUDA runtime support targets enabled to link a launchable executable." >&2
  exit 1
fi

CUDA_LIB_DIR="${CUDA_HOME}/lib64"
if [[ -e "${CUDA_HOME}/lib64/stubs/libcuda.so" ]]; then
  CUDA_STUB_LIB_DIR="${CUDA_HOME}/lib64/stubs"
else
  CUDA_STUB_LIB_DIR="${CUDA_LIB_DIR}"
fi

g++ -I"${CUDA_HOME}/include" -c "${SAMPLE_CALL}" -o sample_call.o
g++ -no-pie sample_call.o sample.o -o a.out \
  -L"${MLIR_BUILD_DIR}/lib" -lmlir_runner_utils -lmlir_cuda_runtime \
  -L"${CUDA_STUB_LIB_DIR}" -L"${CUDA_LIB_DIR}" -lcuda -lcudart \
  -Wl,--wrap=malloc -Wl,--wrap=free \
  -Wl,-rpath,"${MLIR_BUILD_DIR}/lib" \
  -Wl,-rpath,"${CUDA_LIB_DIR}"
