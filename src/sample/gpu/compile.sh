#!/usr/bin/env bash
set -e
set -o pipefail

MLIR_BUILD_DIR="${MLIR_BUILD_DIR:-/build/build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
SAMPLE_CALL="${SAMPLE_CALL:-sample_call.cpp}"
OUTPUT_BINARY="${OUTPUT_BINARY:-a.out}"
SAMPLE_OBJECT="${SAMPLE_OBJECT:-sample.o}"
COMPILE_WORK_DIR="${COMPILE_WORK_DIR:-.}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ ! -f "${SAMPLE_OBJECT}" ]]; then
  echo "${SAMPLE_OBJECT} is missing. Run: bash run_mlir_pipeline.sh" >&2
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

extra_libraries=()
mkdir -p "${COMPILE_WORK_DIR}"
if [[ "${LINK_CUBLAS:-0}" == "1" ]]; then
  extra_libraries+=("-lcublas")
fi
extra_objects=()
if [[ "${LINK_TUTORIAL_GPU_RUNTIME:-0}" == "1" ]]; then
  g++ -std=c++17 -O3 -I"${CUDA_HOME}/include" \
    -I"${PROJECT_ROOT}/src/runtime" \
    -c "${PROJECT_ROOT}/src/runtime/TutorialGpuRuntime.cpp" \
    -o "${COMPILE_WORK_DIR}/tutorial_gpu_runtime.o"
  extra_objects+=("${COMPILE_WORK_DIR}/tutorial_gpu_runtime.o")
  extra_libraries+=("-lcublas")
fi
if [[ "${LINK_CUDNN:-0}" == "1" ]]; then
  if [[ -e "${CUDA_HOME}/include/cudnn.h" ]]; then
    CUDNN_INCLUDE_DIR="${CUDA_HOME}/include"
  elif [[ -e "/usr/include/cudnn.h" ]]; then
    CUDNN_INCLUDE_DIR="/usr/include"
  else
    echo "cuDNN headers not found under ${CUDA_HOME}/include or /usr/include." >&2
    exit 1
  fi
  g++ -std=c++17 -O3 -I"${CUDA_HOME}/include" -I"${CUDNN_INCLUDE_DIR}" \
    -I"${PROJECT_ROOT}/src/runtime" \
    -c "${PROJECT_ROOT}/src/runtime/TutorialCudnnRuntime.cpp" \
    -o "${COMPILE_WORK_DIR}/tutorial_cudnn_runtime.o"
  extra_objects+=("${COMPILE_WORK_DIR}/tutorial_cudnn_runtime.o")
  extra_libraries+=("-lcudnn")
fi
wrap_flags=()
if [[ "${WRAP_MALLOC:-1}" == "1" ]]; then
  wrap_flags+=("-Wl,--wrap=malloc" "-Wl,--wrap=free")
fi

g++ -std=c++17 -O3 -I"${CUDA_HOME}/include" -c "${SAMPLE_CALL}" \
  -o "${COMPILE_WORK_DIR}/sample_call.o"
g++ -no-pie "${COMPILE_WORK_DIR}/sample_call.o" "${SAMPLE_OBJECT}" \
  "${extra_objects[@]}" -o "${OUTPUT_BINARY}" \
  -L"${MLIR_BUILD_DIR}/lib" -lmlir_runner_utils -lmlir_cuda_runtime \
  -L"${CUDA_STUB_LIB_DIR}" -L"${CUDA_LIB_DIR}" -lcuda -lcudart "${extra_libraries[@]}" \
  "${wrap_flags[@]}" \
  -Wl,-rpath,"${MLIR_BUILD_DIR}/lib" \
  -Wl,-rpath,"${CUDA_LIB_DIR}"
