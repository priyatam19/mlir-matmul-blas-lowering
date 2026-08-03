#!/usr/bin/env bash
set -euo pipefail

PROJ="${PROJ:-/workspace/mlir_project}"
BUILD="${BUILD:-${PROJ}/build-ninja}"
MLIR_BUILD_DIR="${MLIR_BUILD_DIR:-/build/build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_CHIP="${CUDA_CHIP:-sm_89}"
CUDA_PTX_FEATURE="${CUDA_PTX_FEATURE:-+ptx80}"
GPU_LOWERING="${GPU_LOWERING:-block-thread}"
MODEL="${PROJ}/src/benchmarks/gpu_residual_conv_block.mlir"
HARNESS="${PROJ}/src/benchmarks/gpu_residual_conv_block_bench.cpp"
ARTIFACT_DIR="${BUILD}/benchmark-artifacts/residual_conv_${GPU_LOWERING}"
BINARY="${ARTIFACT_DIR}/residual_conv.out"

mkdir -p "${ARTIFACT_DIR}"
export PATH="${MLIR_BUILD_DIR}/bin:${BUILD}/tools:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MLIR_BUILD_DIR}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

(
  cd "${PROJ}/src/sample/gpu"
  MODEL_MLIR="${MODEL}" GPU_LOWERING="${GPU_LOWERING}" \
    CUDA_CHIP="${CUDA_CHIP}" CUDA_PTX_FEATURE="${CUDA_PTX_FEATURE}" \
    OUTPUT_DIR="${ARTIFACT_DIR}" bash run_mlir_pipeline.sh
  link_cudnn=0
  if [[ "${GPU_LOWERING}" == "vendor" ]]; then
    link_cudnn=1
  fi
  SAMPLE_CALL="${HARNESS}" SAMPLE_OBJECT="${ARTIFACT_DIR}/sample.o" \
    COMPILE_WORK_DIR="${ARTIFACT_DIR}" OUTPUT_BINARY="${BINARY}" \
    LINK_CUDNN="${link_cudnn}" WRAP_MALLOC=1 bash compile.sh
)

if [[ "${BUILD_ONLY:-0}" == "1" ]]; then
  echo "Built ${BINARY}" >&2
  exit 0
fi

PYTORCH_REFERENCE="${ARTIFACT_DIR}/pytorch_reference.bin"
RESIDUAL_CONV_REFERENCE_OUT="${PYTORCH_REFERENCE}" REFERENCE_ONLY=1 \
  python3 "${PROJ}/src/benchmarks/pytorch_residual_conv_bench.py"
export PYTORCH_REFERENCE

if [[ "${CHECK_LAUNCHES:-0}" == "1" && "${GPU_LOWERING}" == "vendor" ]]; then
  diagnostic_log="$(mktemp)"
  TUTORIAL_GPU_RUNTIME_DEBUG=1 LAUNCH_CHECK_ONLY=1 \
    GPU_LOWERING="${GPU_LOWERING}" "${BINARY}" 2>"${diagnostic_log}"
  call_count="$(grep -c 'op=cudnn_conv2d' "${diagnostic_log}" || true)"
  echo "vendor_call_count=${call_count} expected=2 residual_conv_block" >&2
  rm -f "${diagnostic_log}"
  [[ "${call_count}" == "2" ]]
fi

GPU_LOWERING="${GPU_LOWERING}" WARMUPS="${WARMUPS:-5}" RUNS="${RUNS:-20}" \
  "${BINARY}"
